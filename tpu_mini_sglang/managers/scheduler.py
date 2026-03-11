import logging
import signal
from multiprocessing.connection import Connection

import jax.numpy as jnp
import psutil
import zmq

from tpu_mini_sglang.managers.io_struct import (
    BatchTokenIDOutput,
    TokenizedGenerateRequest,
)
from tpu_mini_sglang.managers.scheduler_struct import (
    GenerationBatchResult,
    ReqState,
    ScheduleBatch,
)
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.models.model_loader import load_model
from tpu_mini_sglang.server_args import PortArgs, ServerArgs
from tpu_mini_sglang.sharding import create_device_mesh
from tpu_mini_sglang.utils import get_exception_traceback, get_zmq_socket

logger = logging.getLogger(__name__)


class Scheduler:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        self.server_args = server_args

        self.model_config = ModelConfig.from_server_args(self.server_args)
        self.max_req_len = self.model_config.context_len

        # Init model
        self.mesh = create_device_mesh(
            data_parallelism=self.server_args.dp, tensor_parallelism=self.server_args.tp
        )
        self.model = load_model(config=self.model_config, mesh=self.mesh)

        # Init ZMQ sockets for IPC
        context = zmq.Context(2)  # Creates 2 io threads
        self.recv_from_tokenizer = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, False
        )
        self.send_to_detokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.detokenizer_ipc_name, False
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

            if self.cur_batch:
                result = self._run_batch(self.cur_batch)
                self._process_batch_result(self.cur_batch, result)

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
        # Simplistic brute force logic to get something that works temporarily
        # Will have much more sophisticated logic once chunked prefill is implemented

        if self.cur_batch is None:
            batch = ScheduleBatch(reqs=[], model_config=self.model_config)
        else:
            batch = self.cur_batch

        batch.reqs = [req for req in batch.reqs if req.finished_reason is None]
        batch.reqs.extend(self.waiting_queue)
        self.waiting_queue = []

        if len(batch.reqs) == 0:
            return None
        else:
            return batch

    def _run_batch(self, batch: ScheduleBatch) -> GenerationBatchResult:
        # Run the forward pass and sampling
        next_token_ids = []
        for req in batch.reqs:
            ids = req.origin_input_ids + req.output_ids

            # Add padding to prevent excessive JAX jits
            # Since we must recompile every time the input is a different size
            CHUNK_SIZE = 256
            pad_len = (CHUNK_SIZE - (len(ids) % CHUNK_SIZE)) % CHUNK_SIZE

            # We pad with 0s to reach the next chunk size
            input = jnp.array(ids + pad_len * [0])
            positions = jnp.concatenate(
                (jnp.arange(len(ids)), jnp.zeros(pad_len, dtype=jnp.int32)), axis=-1
            )

            logits = self.model(input, positions)[len(ids) - 1]
            next_token_ids.append(jnp.argmax(logits).item())  # Greedy sampling for now
        return GenerationBatchResult(next_token_ids=next_token_ids)

    def _process_batch_result(self, batch: ScheduleBatch, result: GenerationBatchResult) -> None:
        """
        First, use the GenerationBatchResult to update batch.reqs.
        Then, use the updated batch.reqs to send BatchTokenIDOutput to the detokenizer.
        """
        for req, next_token_id in zip(batch.reqs, result.next_token_ids, strict=True):
            req.output_ids.append(next_token_id)
            req.check_finished()

        self._stream_output(batch.reqs)

    def _stream_output(self, reqs: list[ReqState]) -> None:
        # Constructs and sends the BatchTokenIDOutput from the requests
        rids = []
        finished_reasons = []
        prompt_ids = []
        output_ids = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []

        for req in reqs:
            rids.append(req.rid)
            finished_reasons.append(req.finished_reason)
            # Also send prompt_ids on the first send
            # This provides necessary context for the detokenizer
            if req.send_token_offset == 0:
                prompt_ids.append(req.origin_input_ids)
            else:
                prompt_ids.append([])
            output_ids.append(req.output_ids[req.send_token_offset :])
            req.send_token_offset = len(req.output_ids)
            prompt_tokens.append(len(req.origin_input_ids))
            completion_tokens.append(len(req.output_ids))
            cached_tokens.append(0)

        output = BatchTokenIDOutput(
            rids=rids,
            finished_reasons=finished_reasons,
            prompt_ids=prompt_ids,
            output_ids=output_ids,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        self.send_to_detokenizer.send_pyobj(output)


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
