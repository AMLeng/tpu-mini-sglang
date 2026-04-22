"""Pydantic models and conversion functions for OpenAI protocol"""

import time
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import Request
from pydantic import BaseModel, Field

from tpu_mini_sglang.managers.io_struct import GenerateRequest, ResponseDict
from tpu_mini_sglang.sampling.sampling_params import UNLIMITED_NEW_TOKENS, SamplingParams

############## Pydantic Models ##############


class ContentPart(BaseModel):
    type: Literal["text", "refusal"]
    text: str


class ChatCompletionDeveloperMessageParam(BaseModel):
    content: str | list[ContentPart]
    role: Literal["developer"]
    name: str | None = None


class ChatCompletionUserMessageParam(BaseModel):
    content: str | list[ContentPart]
    role: Literal["user"]
    name: str | None = None


ChatCompletionMessageParam = ChatCompletionDeveloperMessageParam | ChatCompletionUserMessageParam
# We only support a minimal subset of possible message param types
# See the OpenAI schema for the full list
# https://developers.openai.com/api/reference/resources/chat#(resource)%20chat.completions%20%3E%20(model)%20chat_completion_message_param%20%3E%20(schema)


class CompletionUsage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionRequest(BaseModel):
    # https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create
    # We only include fields that are actually used downstream
    messages: list[ChatCompletionMessageParam]
    model: str
    frequency_penalty: float = 0.0
    logit_bias: dict[str, float] | None = None
    max_completion_tokens: int | None = None
    presence_penalty: float = 0.0
    stream: bool = False
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = -1
    ignore_eos: bool = False


class ChatCompletionMessage(BaseModel):
    content: str | None = None
    refusal: str | None = None
    role: Literal["assistant"] = "assistant"


class ChatCompletionResponseChoice(BaseModel):
    message: ChatCompletionMessage
    finish_reason: Literal["stop", "length", "abort"] | None
    index: int
    # logprobs: None


class ChatCompletion(BaseModel):
    # https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion.py
    id: str
    choices: list[ChatCompletionResponseChoice]
    created: int
    model: str
    object: Literal["chat.completion"] = "chat.completion"
    # service_tier: None
    # system_fingerprint: None
    usage: CompletionUsage | None = None


class ChoiceDelta(BaseModel):
    content: str | None = None
    refusal: str | None = None
    role: Literal["assistant"] | None = None


class ChatCompletionResponseStreamChoice(BaseModel):
    delta: ChoiceDelta
    finish_reason: Literal["stop", "length", "abort"] | None
    index: int
    # logprobs: None


class ChatCompletionChunk(BaseModel):
    # https://github.com/openai/openai-python/blob/main/src/openai/types/chat/chat_completion_chunk.py
    id: str
    choices: list[ChatCompletionResponseStreamChoice]
    created: int
    model: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    # service_tier: None
    # system_fingerprint: None
    # usage: None


class ModelCard(BaseModel):
    """Model cards."""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "tpu-mini-sglang"
    root: str | None
    max_model_len: int | None = None


class ModelList(BaseModel):
    """List of model cards."""

    object: str = "list"
    data: list[ModelCard] = Field(default_factory=list)


class Error(BaseModel):
    message: str
    param: str | None = None
    type: str
    code: str | None = None


class ErrorResponse(BaseModel):
    error: Error


############## Conversion Methods ##############


def make_oai_error_response(error: ValueError) -> str:
    return ErrorResponse(
        error=Error(
            message=str(error),
            type="invalid_request_error",
        ),
    ).model_dump_json()


def convert_chat_completion_to_internal_request(
    req: ChatCompletionRequest, raw_request: Request
) -> GenerateRequest:
    # The raw_request is necessary to access the tokenizer_manager from app.state

    max_new_tokens = (
        req.max_completion_tokens if req.max_completion_tokens is not None else UNLIMITED_NEW_TOKENS
    )
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        frequency_penalty=req.frequency_penalty,
        presence_penalty=req.presence_penalty,
        ignore_eos=req.ignore_eos,
        logit_bias=req.logit_bias if req.logit_bias else {},
    )

    # Pull out any text that lives in content parts
    for message in req.messages:
        if isinstance(message.content, list):
            message.content = "".join(part.text for part in message.content)

    tokenized_templated_input = (
        raw_request.app.state.tokenizer_manager.tokenizer.apply_chat_template(
            req.messages, tokenize=True, add_generation_prompt=True
        )
    )
    return GenerateRequest(
        input_ids=tokenized_templated_input, sampling_params=sampling_params, stream=req.stream
    )


def oai_format_response(response: ResponseDict, model_name: str) -> ChatCompletion:
    rid = response["meta_info"]["id"]
    message = ChatCompletionMessage(
        content=response["text"],
    )
    choice = ChatCompletionResponseChoice(
        message=message,
        finish_reason=response["meta_info"]["finish_reason"],
        index=0,  # We do not support parallel sampling; the index is always 0
    )
    completion_tokens = response["meta_info"]["completion_tokens"]
    prompt_tokens = response["meta_info"]["prompt_tokens"]
    return ChatCompletion(
        id=rid,
        choices=[choice],
        created=int(time.time()),
        model=model_name,
        usage=CompletionUsage(
            completion_tokens=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_tokens=completion_tokens + prompt_tokens,
        ),
    )


async def oai_format_response_stream(
    response_stream: AsyncIterator[ResponseDict], model_name: str
) -> AsyncIterator[str]:
    is_first = True
    finish_reason_type = None
    stream_buffer = ""
    try:
        async for content in response_stream:
            # We do not support parallel sampling; the index is always 0
            index = 0
            if content["meta_info"]["finish_reason"] is not None:
                finish_reason_type = content["meta_info"]["finish_reason"]

            rid = content["meta_info"]["id"]

            # Emit the initial chunk with the role
            if is_first:
                is_first = False
                first_choice = ChatCompletionResponseStreamChoice(
                    delta=ChoiceDelta(role="assistant", content=""),
                    finish_reason=None,
                    index=index,
                )
                first_chunk = ChatCompletionChunk(
                    id=rid, created=int(time.time()), choices=[first_choice], model=model_name
                )
                yield f"data: {first_chunk.model_dump_json()}\n\n"

            delta = content["text"][len(stream_buffer) :]
            stream_buffer = content["text"]

            if delta:
                choice = ChatCompletionResponseStreamChoice(
                    delta=ChoiceDelta(content=delta), finish_reason=None, index=index
                )
                chunk = ChatCompletionChunk(
                    id=rid, created=int(time.time()), choices=[choice], model=model_name
                )
                yield f"data: {chunk.model_dump_json()}\n\n"
        # Emit final chunk, using the id from the last round of the loop
        finish_reason_chunk = ChatCompletionChunk(
            id=rid,
            created=int(time.time()),
            choices=[
                ChatCompletionResponseStreamChoice(
                    delta=ChoiceDelta(), finish_reason=finish_reason_type, index=index
                )
            ],
            model=model_name,
        )
        yield f"data: {finish_reason_chunk.model_dump_json()}\n\n"
    except ValueError as e:
        error_response = ErrorResponse(
            error=Error(
                message=str(e),
                type="invalid_request_error",
            ),
        )
        yield f"data: {error_response.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
