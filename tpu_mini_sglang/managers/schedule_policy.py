from enum import Enum, auto

from tpu_mini_sglang.managers.scheduler_struct import (
    PrefillReqState,
    PreparedReqState,
    ReqInfo,
    ScheduleBatch,
)


class AddReqResult(Enum):
    CONTINUE = auto()  # Continue to add requests
    NO_TOKEN = auto()  # No KV cache slots left
    OTHER = auto()  # Other reasons to stop adding requests


class PrefillAdder:
    def __init__(
        self,
        page_size: int,
        available_kv_tokens: int,
        running_decode_batch: ScheduleBatch | None,
        prefill_token_budget: int,
        available_req_slots: int,
    ):
        self.page_size = page_size

        self.rem_kv_tokens = available_kv_tokens
        # Adjust kv budget to reserve capacity for all currently running decode batches
        if running_decode_batch is not None:
            self.rem_kv_tokens -= sum(
                r.req_info.sampling_params.max_new_tokens - len(r.output_ids)
                for r in running_decode_batch.reqs
            )
        self.rem_token_budget = prefill_token_budget

        self.available_req_slots = available_req_slots
        # A chunked req will appear in can_run_list but will already have a req slot
        # So we track how many chunked reqs are present
        self.chunked_reqs_in_can_run_list = 0

        self.can_run_list: list[PrefillReqState] = []

    def _ceil_paged_tokens(self, tokens: int):
        return -(-tokens // self.page_size) * self.page_size

    def _update_prefill_budget(self, extend_len: int, max_new_tokens: int) -> None:
        extend_len = self._ceil_paged_tokens(extend_len)

        self.rem_kv_tokens -= max_new_tokens + extend_len
        self.rem_token_budget -= extend_len

    def try_add_chunked_req(self, chunked_req: PreparedReqState) -> None | PreparedReqState:
        # We will ultimately call the tree cache to match here, but for now we just use the
        # extra chunked_prefix_len we saved
        prefix_len = chunked_req.chunked_prefix_len + chunked_req.extend_len

        # Add as much of a chunked req as possible, returning the req if not fully added
        # SGLang unconditionally adds the req, potentially overcommitting the KV cache to do so
        # we instead allow for not running the req and thus letting the decode batch make progress
        _rem_tokens = min(self.rem_token_budget, self.rem_kv_tokens)
        full_extend_len = len(chunked_req.req_info.origin_input_ids) - prefix_len
        tokens_to_add = min(_rem_tokens, full_extend_len)

        if tokens_to_add <= 0:
            # We don't have any free cache so just keep this as the chunked req and return
            # Different from original SGLang, we only return chunked_req on this early-return
            # path which doesn't exist in the original SGLang. For us, chunked reqs that enter
            # the can_run_list will be processed after running the forward batch
            return chunked_req

        truncated = tokens_to_add < full_extend_len
        req = PrefillReqState(
            req_info=chunked_req.req_info,
            prefix_len=prefix_len,
            extend_len=tokens_to_add,
            req_pool_idx=chunked_req.req_pool_idx,
            prefill_unfinished=truncated,
        )
        self.can_run_list.append(req)
        self.chunked_reqs_in_can_run_list += 1
        # When truncating, we allocate budget for the next chunked prefill stage and not for decode
        self._update_prefill_budget(
            tokens_to_add,
            req.req_info.sampling_params.max_new_tokens if not truncated else 0,
        )
        return None

    def try_add_one_req(self, req_info: ReqInfo) -> AddReqResult:
        # We will ultimately call the tree cache to match here, but for now we just use the
        # knowledge that without any prefix matching, the prefix len will always be 0
        prefix_len = 0
        extend_len = len(req_info.origin_input_ids) - prefix_len

        if self.available_req_slots + self.chunked_reqs_in_can_run_list <= len(self.can_run_list):
            return AddReqResult.OTHER
        if extend_len + req_info.sampling_params.max_new_tokens > self.rem_kv_tokens:
            return AddReqResult.NO_TOKEN
        # Make sure at least one page is available
        if self.rem_token_budget < self.page_size:
            return AddReqResult.OTHER

        tokens_to_add = min(self.rem_token_budget, extend_len)

        # If extend_len is larger, prefill won't finish so we will chunk the req
        prefill_unfinished = tokens_to_add < extend_len

        req = PrefillReqState(
            req_info=req_info,
            prefix_len=prefix_len,
            extend_len=tokens_to_add,
            prefill_unfinished=prefill_unfinished,
        )

        self.can_run_list.append(req)
        self._update_prefill_budget(
            tokens_to_add,
            req_info.sampling_params.max_new_tokens if not prefill_unfinished else 0,
        )
        return AddReqResult.CONTINUE

    def filter_runnable_reqs(self, waiting_list: list[ReqInfo]):
        rids = {r.req_info.rid for r in self.can_run_list}
        return [r for r in waiting_list if r.rid not in rids]
