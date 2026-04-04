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
from tpu_mini_sglang.managers.schedule_policy import (
    AddReqResult,
    PrefillAdder,
)
from tpu_mini_sglang.managers.scheduler_struct import (
    GenerationBatchResult,
    PrefillReqState,
    ProcessedReqState,
    ReqInfo,
    ScheduleBatch,
)
from tpu_mini_sglang.mem_cache.allocator import TokenToKVPoolAllocator
from tpu_mini_sglang.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from tpu_mini_sglang.mem_cache.radix_cache import RadixCache
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
        self.waiting_queue: list[ReqInfo] = []
        self.running_decode_batch: ScheduleBatch | None = None
        self.chunked_req: PrefillReqState | None = None

        # Init model runner
        self.mesh = create_device_mesh(
            data_parallelism=self.server_args.dp, tensor_parallelism=self.server_args.tp
        )
        self.model_runner = ModelRunner(
            self.model_config,
            self.mesh,
        )

        # Init KV Cache
        assert self.server_args.page_size == 1  # The allocator cannot handle other page sizes yet
        self._init_memory_pool_and_cache(
            max_kv_tokens=self.model_runner.get_max_kv_tokens(self.model_config.dtype),
            page_size=self.server_args.page_size,
            kv_cache_dtype=self.model_config.dtype,
        )

    def run_event_loop(self):
        while True:
            recv_reqs = self._recv_requests()

            # Any generate requests will be processed and stored in self.waiting_queue
            self._process_input_requests(recv_reqs)

            # Form batches from requests in self.waiting_queue
            cur_batch = self._get_next_batch_to_run()

            if cur_batch:
                result = self._run_batch(cur_batch)
                new_decode_batch = self._process_batch_result(cur_batch, result)
                if new_decode_batch is None:
                    continue
                if self.running_decode_batch:
                    # Both batches we merge are prepared and ready to run
                    self.running_decode_batch.merge_batch(new_decode_batch)
                else:
                    self.running_decode_batch = new_decode_batch

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
        """Create ReqInfo for the request and add it to the queue."""
        # If the user didn't provide max_new_tokens, it is initialized to a massive default
        # We clamp it here, since the scheduler is the source of truth for max_req_len
        sampling_params = recv_req.sampling_params
        sampling_params.max_new_tokens = min(
            sampling_params.max_new_tokens,
            self.max_req_len - len(recv_req.input_ids),
        )
        req_info = ReqInfo(
            rid=recv_req.rid,
            origin_input_ids=recv_req.input_ids,
            sampling_params=sampling_params,
            eos_token_ids=self.model_config.hf_eos_token_id,
        )
        self.waiting_queue.append(req_info)

    def _get_next_batch_to_run(self) -> ScheduleBatch | None:
        adder = PrefillAdder(
            page_size=self.server_args.page_size,
            tree_cache=self.tree_cache,
            available_kv_tokens=self.token_to_kv_pool_allocator.available_size(),
            running_decode_batch=self.running_decode_batch,
            prefill_token_budget=self.server_args.max_num_batched_tokens,
            available_req_slots=self.req_to_token_pool.available_size(),
        )

        if self.chunked_req is not None:
            # Will only return here if we cannot make any progress on the chunked req right now
            self.chunked_req = adder.try_add_chunked_req(self.chunked_req)

        for req in self.waiting_queue:
            res = adder.try_add_one_req(req)
            if res != AddReqResult.CONTINUE:
                break

        self.waiting_queue = adder.filter_runnable_reqs(self.waiting_queue)

        # Mimic SGLang behavior and run prefill if we have a prefill batch
        if len(adder.can_run_list) > 0:
            return ScheduleBatch.prepare_for_prefill(
                reqs=adder.can_run_list,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                tree_cache=self.tree_cache,
            )
        # Else run decode
        if self.running_decode_batch:
            decode_batch = self.running_decode_batch
            self.running_decode_batch = None
            return decode_batch
        return None

    def _run_batch(self, batch: ScheduleBatch) -> GenerationBatchResult:
        return self.model_runner.forward_batch_generation(self.kv_cache, batch)

    def _process_batch_result(
        self, batch: ScheduleBatch, result: GenerationBatchResult
    ) -> ScheduleBatch | None:
        """
        First, use the GenerationBatchResult to update batch.reqs.
        Then, use the updated batch.reqs to send BatchTokenIDOutput to the detokenizer.
        Finally, remove finished requests and free kv cache
        """

        # Pull out and save the chunked req if present
        chunked_reqs = [r for r in batch.reqs if r.prefill_unfinished]
        assert len(chunked_reqs) + int(self.chunked_req is not None) <= 1
        if len(chunked_reqs) == 1:
            self.chunked_req = self.tree_cache.cache_chunked_req(chunked_reqs[0])

        # Then begin processing all non-chunked reqs
        # Past this point, we should not reference batch any more
        reqs = [
            ProcessedReqState.process_req(r, next_token_id)
            for r, next_token_id in zip(batch.reqs, result.next_token_ids, strict=True)
            if not r.prefill_unfinished
        ]

        self._stream_output(reqs)

        # Update caches
        finished_reqs = [r for r in reqs if r.finished_reason is not None]
        unfinished_reqs = [r for r in reqs if r.finished_reason is None]
        req_pool_indices = []
        for req in finished_reqs:
            self.tree_cache.cache_finished_req(req)
            req_pool_indices.append(req.req_pool_idx)
        self.req_to_token_pool.free(req_pool_indices)  # Must happen after we cache reqs
        for req in unfinished_reqs:
            self.tree_cache.cache_unfinished_req(req)

        if len(unfinished_reqs) == 0:
            return None
        return ScheduleBatch.prepare_for_decode(
            reqs=unfinished_reqs,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            tree_cache=self.tree_cache,
        )

    def _stream_output(self, reqs: list[ProcessedReqState]) -> None:
        # Constructs and sends the BatchTokenIDOutput from the requests
        rids = []
        finished_reasons = []
        prompt_ids = []
        output_ids = []
        prompt_tokens = []
        completion_tokens = []
        cached_tokens = []

        for req in reqs:
            rids.append(req.req_info.rid)
            finished_reasons.append(req.finished_reason)
            # Also send prompt_ids on the first send
            # This provides necessary context for the detokenizer
            if req.send_token_offset == 0:
                prompt_ids.append(req.req_info.origin_input_ids)
            else:
                prompt_ids.append([])
            output_ids.append(req.output_ids[req.send_token_offset :])
            req.send_token_offset = len(req.output_ids)
            prompt_tokens.append(len(req.req_info.origin_input_ids))
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

    def _init_memory_pool_and_cache(
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

        self.tree_cache = RadixCache(
            page_size=page_size,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
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
