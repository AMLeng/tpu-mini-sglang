import logging
import signal
from multiprocessing.connection import Connection

import psutil
import zmq

from tpu_mini_sglang.managers.io_struct import (
    TokenizedGenerateRequest,
)
from tpu_mini_sglang.managers.scheduler_struct import (
    GenerationBatchResult,
    ReqState,
    ScheduleBatch,
)
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.server_args import PortArgs, ServerArgs
from tpu_mini_sglang.utils import get_exception_traceback, get_zmq_socket

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args

        self.model_config = ModelConfig.from_server_args(self.server_args)
        self.max_req_len = self.model_config.context_len

        # Init ZMQ sockets for IPC
        context = zmq.Context(2)  # Creates 2 io threads
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, False
        )
        self.send_to_detokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.detokenizer_ipc_name, True
        )

        # Init running state
        self.waiting_queue: list[ReqState] = []
        self.cur_batch: ScheduleBatch | None = None

    def run_event_loop(self):
        while True:
            recv_reqs = self._recv_requests()

            # Any generate requests will be processed and stored in self.waiting_queue
            self._process_input_requests(recv_reqs)

            # Form batches from requests in self.waiting_queue
            self.cur_batch = self._get_next_batch_to_run()

            # if self.cur_batch:
            # result = self._run_batch(self.cur_batch)
            # self._process_batch_result(result)

    def _recv_requests(self):
        """Read from ZMQ socket until there is nothing left"""
        reqs = []
        while True:
            try:
                # with NOBLOCK, throws ZMQError if nothing to read
                req = self.recv_from_tokenizer.recv_pyobj(zmq.NOBLOCK)
            except zmq.ZMQError:
                break
            reqs.append(req)
        return reqs

    def _process_input_requests(self, recv_reqs):
        """Parses and prepares input requests on the CPU."""
        for recv_req in recv_reqs:
            if isinstance(recv_req, TokenizedGenerateRequest):
                self._handle_generate_request(recv_req)
            else:
                raise ValueError(f"Invalid object: {recv_req}")

    def _handle_generate_request(self, recv_req: TokenizedGenerateRequest):
        """Create ReqState for the request and add it to the queue."""
        req_state = ReqState(
            rid=recv_req.rid,
            origin_input_ids=recv_req.input_ids,
            sampling_params=recv_req.sampling_params,
            eos_token_ids=self.model_config.hf_eos_token_id,
            vocab_size=self.model_config.vocab_size,
        )
        # Mutation safe since this copy of sampling_params was deserialized with recv_req
        req_state.sampling_params.max_new_tokens = min(
            (
                req_state.sampling_params.max_new_tokens
                if req_state.sampling_params.max_new_tokens is not None
                else 1 << 30
            ),
            self.max_req_len - len(req_state.origin_input_ids),
        )
        self.waiting_queue.append(req_state)

    def _get_next_batch_to_run(self) -> ScheduleBatch | None:
        return None

    # def _run_batch(self, batch: ScheduleBatch) -> GenerationBatchResult:
    # Run the forward pass and sampling on the device

    def _process_batch_result(self, result: GenerationBatchResult) -> None:
        pass


def run_scheduler_process(server_args: ServerArgs, port_args: PortArgs, pipe_writer: Connection):
    parent = psutil.Process().parent()
    try:
        scheduler = Scheduler(server_args, port_args)
        # Magic number of 5 borrowed from original SGLang tp_worker.py
        scheduler_data = {"status": "ready", "max_req_input_len": scheduler.max_req_len - 5}
        pipe_writer.send(scheduler_data)

        scheduler.run_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        if parent is not None:
            parent.send_signal(signal.SIGQUIT)
