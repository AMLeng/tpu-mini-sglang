from collections.abc import AsyncIterator

import orjson
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from tpu_mini_sglang.entrypoints.openai_protocol import (
    ChatCompletionRequest,
    ModelCard,
    ModelList,
    convert_chat_completion_to_internal_request,
    oai_format_response_stream,
)
from tpu_mini_sglang.managers.io_struct import GenerateRequest

app = FastAPI()


@app.get("/health")
async def health() -> Response:
    """Check the health of the HTTP server."""
    return Response(status_code=200)


@app.post("/generate")
async def generate_request(req: GenerateRequest) -> Response:
    """Handle a generate request in a streaming fashion."""

    async def stream_results() -> AsyncIterator[bytes]:
        """Stream the response back to the client as Server-Sent Events (SSE)"""

        async for out in app.state.tokenizer_manager.generate_request_stream(req):
            yield (b"data: " + orjson.dumps(out, option=orjson.OPT_NON_STR_KEYS) + b"\n\n")
        yield (b"data: [DONE]\n\n")

    return StreamingResponse(stream_results(), media_type="text/event-stream")


# Minimal Set of OpenAI Compatible Endpoints
@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(req: ChatCompletionRequest, raw_request: Request):
    """OpenAI style chat completions endpoint."""
    internal_request = convert_chat_completion_to_internal_request(req, raw_request)

    return StreamingResponse(
        oai_format_response_stream(
            app.state.tokenizer_manager.generate_request_stream(internal_request), req.model
        ),
        media_type="text/event-stream",
    )


@app.get("/v1/models")
async def openai_v1_models():
    """List available models. OpenAI-compatible endpoint."""
    model_card = ModelCard(
        id=app.state.tokenizer_manager.served_model_name,
        root=app.state.tokenizer_manager.served_model_name,
        max_model_len=app.state.tokenizer_manager.model_config.context_len,
    )
    return ModelList(data=[model_card])
