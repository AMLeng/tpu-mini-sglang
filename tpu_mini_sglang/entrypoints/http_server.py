import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import orjson
import requests
import uvicorn
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
from tpu_mini_sglang.server_args import ServerArgs
from tpu_mini_sglang.utils import get_exception_traceback, kill_process_tree

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(fast_api_app: FastAPI):

    warmup_thread = threading.Thread(
        target=_execute_server_warmup, args=(fast_api_app.state.server_args,)
    )
    warmup_thread.start()
    try:
        yield
    finally:
        warmup_thread.join()


app = FastAPI(lifespan=lifespan)


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


def launch_server(server_args: ServerArgs):
    """
    Launches the full server, with a HTTP server and an engine.
    The engine spawns/owns the tokenizer_manager, scheduler, and detokenizer_manager.
    Following SGLang, the server, tokenizer_manager, and engine all run in the main process.
    The scheduler and detokenizer_manager run as separate subprocesses.
    """

    # Save the server args so they can be accessed by the warmup
    app.state.server_args = server_args

    uvicorn.run(
        app,
        host=server_args.host,
        port=server_args.port,
        log_level=server_args.log_level,
        timeout_keep_alive=5,
        loop="uvloop",
    )


def _execute_server_warmup(server_args: ServerArgs):
    if server_args.skip_server_warmup:
        return True

    url = server_args.url

    # Wait until the server is launched
    success = False
    last_traceback = ""
    for _ in range(120):
        time.sleep(1)
        try:
            res = requests.get(url + "/health", timeout=5)
            res.raise_for_status()
            success = True
            break
        except requests.exceptions.RequestException:
            last_traceback = get_exception_traceback()
            pass
    if not success:
        logger.error("Initialization failed. Warmup error: %s", last_traceback)
        kill_process_tree()
        return success
    request_name = "/generate"
    max_new_tokens = 8
    warmup_timeout = 600

    # Unlike the original SGLang, our /generate input only takes in a single string, not an array
    # Hence json_data["text"] is always just a string
    json_data = {
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens},
        "text": "The capital city of France is",
    }

    try:
        res = requests.post(url + request_name, json=json_data, timeout=warmup_timeout)
        res.raise_for_status()
    except Exception:
        last_traceback = get_exception_traceback()
        logger.error("Initialization failed. Warmup error: %s", last_traceback)
        kill_process_tree()
        return False

    return success
