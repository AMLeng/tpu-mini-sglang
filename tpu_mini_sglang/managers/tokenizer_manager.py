import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from transformers import AutoTokenizer

from tpu_mini_sglang.managers.io_struct import (
    GenerateRequest,
    ResponseDict,
    TokenizedGenerateRequest,
)
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.server_args import ServerArgs


@dataclass
class ReqState:
    latest_out: ResponseDict | None
    finished: bool
    event: asyncio.Event
    text: str = ""
    output_ids: list[int] = field(default_factory=list)


class TokenizerManager:
    """
    Tokenizes input text and manages the request lifecycle between the
    HTTP server and the scheduler/detokenizer processes. Tracks in-flight
    request state, coordinates async streaming responses, and handles
    IPC over ZMQ.
    """

    def __init__(self, server_args: ServerArgs):
        self.server_args = server_args

        # Init model config
        self.model_path = server_args.model_path
        self.model_config = ModelConfig.from_server_args(server_args)

        # Init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(server_args.model_path)

        # Init IPC

        # Init running status
        self.rid_to_state: dict[str, ReqState] = {}

        # Init request dispatcher

    async def generate_request_stream(self, req: GenerateRequest) -> AsyncIterator[ResponseDict]:
        # Tokenize the request text if not already tokenized
        tokenized_req = self._tokenize_one_request(req)

        # Send request to scheduler
        state = self._send_one_request(tokenized_req)

        # Await the scheduler response
        async for response in self._wait_one_response(req, state):
            yield response

    def _tokenize_one_request(self, obj: GenerateRequest) -> TokenizedGenerateRequest:
        if obj.input_ids is not None:
            input_ids = obj.input_ids
        else:
            input_ids = self.tokenizer(obj.text)["input_ids"]
        return TokenizedGenerateRequest(
            rid=obj.rid, input_ids=input_ids, sampling_params=obj.sampling_params
        )

    def _send_one_request(self, tokenized_req: TokenizedGenerateRequest) -> ReqState:
        # self.send_to_scheduler.send_pyobj(tokenized_req)
        state = ReqState(latest_out=None, finished=False, event=asyncio.Event())
        self.rid_to_state[tokenized_req.rid] = state
        return state

    async def _wait_one_response(
        self, req: GenerateRequest, state: ReqState
    ) -> AsyncIterator[ResponseDict]:
        # state.event will be set after state.latest_out is populated by the detokenizer
        while True:
            await state.event.wait()
            assert state.latest_out is not None
            out = state.latest_out
            state.latest_out = None

            if not state.finished:
                state.event.clear()
                yield out
                continue

            yield out
            break
