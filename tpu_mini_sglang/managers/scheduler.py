import logging
import signal
from multiprocessing.connection import Connection

import numpy as np
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
from tpu_mini_sglang.mem_cache.allocator import TokenToKVPoolAllocator
from tpu_mini_sglang.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from tpu_mini_sglang.model_config import ModelConfig
from tpu_mini_sglang.model_executor.model_runner import ModelRunner
from tpu_mini_sglang.server_args import PortArgs, ServerArgs
from tpu_mini_sglang.sharding import create_device_mesh
from tpu_mini_sglang.utils import configure_logger, get_exception_traceback, get_zmq_socket

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
            context, zmq.PUSH, port_args.detokenizer_ipc_name, False
        )

        # Init running state
        self.waiting_queue: list[ReqState] = []
        self.cur_batch: ScheduleBatch | None = None

        # Init model runner
        self.mesh = create_device_mesh(
            data_parallelism=self.server_args.dp, tensor_parallelism=self.server_args.tp
        )
        self.model_runner = ModelRunner(
            self.model_config,
            self.mesh,
        )

        # Init KV Cache
        self._init_memory_pool(
            max_kv_tokens=self.model_runner.get_max_kv_tokens(self.model_config.dtype),
            page_size=1,
            kv_cache_dtype=self.model_config.dtype,
        )

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
        req_state = ReqState.init_new(
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

        can_run_list = []
        if self.cur_batch:
            can_run_list = self.cur_batch.reqs
        for req in self.waiting_queue:
            if self.req_to_token_pool.available_size() <= 0:
                break

            # Requests from the waiting queue will have nothing in the KV cache
            # We could use more intelligent logic here
            # For now we use the conservative strategy to avoid dealing with cache evictions
            expected_tokens = req.sampling_params.max_new_tokens
            if self.token_to_kv_pool_allocator.available_size() < expected_tokens:
                continue
            can_run_list.append(req)

        self.waiting_queue = [req for req in self.waiting_queue if req not in can_run_list]

        if len(can_run_list) <= 0:
            return None

        batch = ScheduleBatch.init_new(
            reqs=can_run_list,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            model_config=self.model_config,
        )

        batch.prepare()
        return batch

    def _run_batch(self, batch: ScheduleBatch) -> GenerationBatchResult:
        return self.model_runner.forward_batch_generation(self.kv_cache, batch)

    def _process_batch_result(self, batch: ScheduleBatch, result: GenerationBatchResult) -> None:
        """
        First, use the GenerationBatchResult to update batch.reqs.
        Then, use the updated batch.reqs to send BatchTokenIDOutput to the detokenizer.
        Finally, remove finished requests and free kv cache
        """
        for req, next_token_id in zip(batch.reqs, result.next_token_ids, strict=True):
            req.output_ids.append(next_token_id)
            req.check_finished()

        self._stream_output(batch.reqs)

        # Free caches and filter batch
        finished_reqs = [r for r in batch.reqs if r.finished_reason is not None]
        req_pool_indices = []
        for req in finished_reqs:
            assert req.req_pool_idx is not None
            self.token_to_kv_pool_allocator.free(
                self.req_to_token_pool.read(
                    req.req_pool_idx, len(req.origin_input_ids) + len(req.output_ids)
                )
            )
            req_pool_indices.append(req.req_pool_idx)
        self.req_to_token_pool.free(req_pool_indices)

        batch.reqs = [req for req in batch.reqs if req.finished_reason is None]

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

    def _init_memory_pool(
        self,
        max_kv_tokens: int,
        page_size: int,
        kv_cache_dtype: np.dtype,
    ):
        # Formula from SGLang
        max_running_requests = min(
            max(
                int(max_kv_tokens / self.model_config.context_len * 512),
                2048,
            ),
            4096,
        )

        self.req_to_token_pool = ReqToTokenPool(
            max_running_requests=max_running_requests, max_context_len=self.model_config.context_len
        )
        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
            size=max_kv_tokens,
        )
        self.kv_cache = MHATokenToKVPool(
            cache_size=max_kv_tokens,
            page_size=page_size,
            num_layers=self.model_config.num_layers,
            num_kv_heads=self.model_config.num_kv_heads,
            head_dim=self.model_config.head_dim,
            mesh=self.mesh,
            dtype=kv_cache_dtype,
        )


def run_scheduler_process(server_args: ServerArgs, port_args: PortArgs, pipe_writer: Connection):
    configure_logger(server_args, prefix="Scheduler:")
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
