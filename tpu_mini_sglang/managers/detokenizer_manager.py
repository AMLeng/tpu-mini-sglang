import logging
import signal
from dataclasses import dataclass

import psutil
import zmq
from tokenizers.decoders import DecodeStream  # type: ignore[import-untyped]
from transformers import AutoTokenizer

from tpu_mini_sglang.managers.io_struct import (
    BatchStrOutput,
    BatchTokenIDOutput,
)
from tpu_mini_sglang.server_args import PortArgs, ServerArgs
from tpu_mini_sglang.utils import configure_logger, get_exception_traceback, get_zmq_socket

logger = logging.getLogger(__name__)


@dataclass
class DecodeStatus:
    """
    Class to track decode information.
    The ids to be decoded are stored within decode_stream.
    """

    # Tracks ids to be decoded and produces checks of decoded text
    decode_stream: DecodeStream
    decoded_text: str = ""  # full value of decoded text
    sent_offset: int = 0  # how much of decoded_text has been sent


class DetokenizerManager:
    def __init__(self, server_args: ServerArgs, port_args: PortArgs):
        # Init (de)tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(server_args.model_path)

        # Init ZMQ sockets for IPC
        context = zmq.Context(2)  # Creates 2 io threads
        self.recv_from_scheduler = get_zmq_socket(
            context, zmq.PULL, port_args.detokenizer_ipc_name, True
        )
        self.send_to_tokenizer = get_zmq_socket(
            context, zmq.PUSH, port_args.tokenizer_ipc_name, False
        )

        # Init running status
        self.rid_to_status: dict[str, DecodeStatus] = {}

    def run_event_loop(self):
        """Core synchronous event loop to handle detokenization requests"""
        while True:
            recv_obj = self.recv_from_scheduler.recv_pyobj()

            if isinstance(recv_obj, BatchTokenIDOutput):
                output = self._handle_batch_token_id_output(recv_obj)
            else:
                raise ValueError(f"Invalid object: {recv_obj}")

            # Send the output to the tokenizer_manager
            # which lives in the main process
            self.send_to_tokenizer.send_pyobj(output)

    def _handle_batch_token_id_output(self, recv_obj: BatchTokenIDOutput) -> BatchStrOutput:
        output_strs = self._decode_batch_token_id_output(recv_obj)
        return BatchStrOutput(
            rids=recv_obj.rids,
            finished_reasons=recv_obj.finished_reasons,
            output_strs=output_strs,
            output_ids=recv_obj.output_ids,
            prompt_tokens=recv_obj.prompt_tokens,
            completion_tokens=recv_obj.completion_tokens,
            cached_tokens=recv_obj.cached_tokens,
        )

    def _decode_batch_token_id_output(self, recv_obj: BatchTokenIDOutput) -> list[str]:
        """
        Instead of using SGLang's old decoder implementation, we borrow vLLM's updated approach.
        Using DecodeStream from tokenizer we decode each request individually token-by-token.
        This greatly simplifies the decoding logic.
        """
        output_strs = []
        for i, rid in enumerate(recv_obj.rids):
            if rid not in self.rid_to_status:
                status = DecodeStatus(
                    decode_stream=DecodeStream(ids=recv_obj.prompt_ids[i], skip_special_tokens=True)
                )
                self.rid_to_status[rid] = status
            else:
                status = self.rid_to_status[rid]

            for token_id in recv_obj.output_ids[i]:
                # We must get the underlying Tokenizer for the tokenizers library
                # Rather than the higher level object from transformers that wraps it
                chunk = status.decode_stream.step(self.tokenizer._tokenizer, token_id)
                if chunk is not None:
                    status.decoded_text += chunk

            if recv_obj.finished_reasons[i] is not None:
                del self.rid_to_status[rid]

            incremental_output = status.decoded_text[status.sent_offset :]
            status.sent_offset = len(status.decoded_text)

            output_strs.append(incremental_output)
        return output_strs


def run_detokenizer_process(server_args: ServerArgs, port_args: PortArgs):
    configure_logger(server_args, prefix="DetokenizerManager:")
    parent = psutil.Process().parent()
    try:
        manager = DetokenizerManager(server_args, port_args)
        manager.run_event_loop()
    except Exception:
        traceback = get_exception_traceback()
        logger.error("DetokenizerManager hit an exception: %s", traceback)
        if parent is not None:
            parent.send_signal(signal.SIGQUIT)
