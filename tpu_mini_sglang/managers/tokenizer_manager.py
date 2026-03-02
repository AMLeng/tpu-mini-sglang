import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import uvloop
import zmq
import zmq.asyncio
from transformers import AutoTokenizer

from tpu_mini_sglang.managers.io_struct import (
    BatchStrOutput,
    GenerateRequest,
    ResponseDict,
    ResponseMetadataDict,
    TokenizedGenerateRequest,
)
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.server_args import PortArgs, ServerArgs
from tpu_mini_sglang.utils import get_or_create_event_loop, get_zmq_socket

asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

logger = logging.getLogger(__name__)


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

    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args

        # Init model config
        self.model_path = server_args.model_path
        self.model_config = ModelConfig.from_server_args(server_args)

        # Init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(server_args.model_path)

        # Init ZMQ sockets for IPC
        context = zmq.asyncio.Context(2)  # Creates 2 io threads
        self.recv_from_detokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.tokenizer_ipc_name, True
        )
        self.send_to_scheduler = get_zmq_socket(
            context, zmq.PUSH, port_args.scheduler_input_ipc_name, True
        )

        # Init running status
        self.rid_to_state: dict[str, ReqState] = {}
        self.event_loop: asyncio.AbstractEventLoop | None = None
        self.asyncio_tasks: set[asyncio.Task] = set()  # References to prevent garbage collection

    async def generate_request_stream(self, req: GenerateRequest) -> AsyncIterator[ResponseDict]:
        # Attach our handler loop to the main event loop the first time we process a request
        # This will allow us to receive the result from the detokenizer
        self.ensure_event_loop()

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
        # Synchronous send on async socket; safe because we have unlimited high water mark
        self.send_to_scheduler.send_pyobj(tokenized_req)

        state = ReqState(latest_out=None, finished=False, event=asyncio.Event())
        self.rid_to_state[tokenized_req.rid] = state
        return state

    async def _wait_one_response(
        self, req: GenerateRequest, state: ReqState
    ) -> AsyncIterator[ResponseDict]:
        # state.event will be set after state.latest_out is populated by _handle_batch_output
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

    def ensure_event_loop(self):
        """
        The tokenizer manager runs in the main process but doesn't own the event loop.
        Thus, we wait for e.g. uvicorn to create the event loop.
        We then attach handle_loop to it.
        """
        if self.event_loop is not None:
            return

        self.event_loop = get_or_create_event_loop()
        self.asyncio_tasks.add(self.event_loop.create_task(self.handle_loop()))

    async def handle_loop(self):
        while True:
            recv_obj = await self.recv_from_detokenizer.recv_pyobj()

            # Match object type to correct handling function
            if isinstance(recv_obj, BatchStrOutput):
                self._handle_batch_output(recv_obj)
            else:
                raise ValueError(f"Invalid object: {recv_obj}")

    def _handle_batch_output(self, recv_obj: BatchStrOutput):
        for i, rid in enumerate(recv_obj.rids):
            state = self.rid_to_state.get(rid, None)
            if state is None:
                logger.error(
                    "Received output for rid=%s but the state was deleted in TokenizerManager.", rid
                )
                continue
            meta_info = ResponseMetadataDict(
                id=rid,
                finish_reason=recv_obj.finished_reasons[i],
                prompt_tokens=recv_obj.prompt_tokens[i],
                completion_tokens=recv_obj.completion_tokens[i],
                cached_tokens=recv_obj.cached_tokens[i],
            )
            state.text += recv_obj.output_strs[i]

            state.output_ids.extend(recv_obj.output_ids[i])

            out_dict = ResponseDict(
                text=state.text, output_ids=recv_obj.output_ids[i], meta_info=meta_info
            )

            state.finished = recv_obj.finished_reasons[i] is not None
            if state.finished:
                # state is removed from the dict
                # but there is still a reference from _wait_one_response
                del self.rid_to_state[rid]

            state.latest_out = out_dict
            state.event.set()
