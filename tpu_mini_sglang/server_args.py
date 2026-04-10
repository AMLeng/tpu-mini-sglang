import dataclasses
import tempfile

from simple_parsing import ArgumentParser, DashVariant


@dataclasses.dataclass
class ServerArgs:
    # The path to the model weights.
    # This can be a local folder or a Hugging Face repo ID.
    model_path: str

    # Sharding

    tp: int = 1  # Degree of tensor parallelism
    dp: int = 1  # Degree of data parallelism

    # HTTP Server

    host: str = "127.0.0.1"  # The host of the HTTP server.
    port: int = 30000  # The port of the HTTP server.
    skip_server_warmup: bool = False  # If set, skip warmup.

    # Logging

    log_level: str = "info"  # The default logging level of all loggers.
    perf_log_level: str = "info"  # The logging level of performance loggers.

    # Scheduling

    page_size: int = 1
    max_num_batched_tokens: int = 8192
    max_num_batched_requests: int = 256
    skip_scheduler_warmup: bool = False  # If set, skip JIT precompilation/Scheduler warmup

    @classmethod
    def build_parser(cls, dest: str = "cfg") -> ArgumentParser:
        parser = ArgumentParser(add_option_string_dash_variants=DashVariant.DASH)
        parser.add_arguments(cls, dest=dest)
        return parser

    @property
    def url(self):
        # Assumes we are using IPv4
        return f"http://{self.host}:{self.port}"


@dataclasses.dataclass
class PortArgs:
    # With multiple hosts, we will have ports rather than ipc socket names
    # ipc filename for detokenizer to send to tokenizer
    tokenizer_ipc_name: str
    # ipc filename for tokenizer to send to scheduler
    scheduler_input_ipc_name: str
    # ipc filename for scheduler to send to detokenizer
    detokenizer_ipc_name: str

    @staticmethod
    def init_new():
        return PortArgs(
            tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
            detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
        )
