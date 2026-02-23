import dataclasses
from typing import Self

import simple_parsing


@dataclasses.dataclass
class ServerArgs:
    # The path to the model weights.
    # This can be a local folder or a Hugging Face repo ID.
    model_path: str

    # HTTP Server

    host: str = "127.0.0.1"  # The host of the HTTP server.
    port: int = 30000  # The port of the HTTP server.
    skip_server_warmup: bool = False  # If set, skip warmup.

    # Logging

    log_level: str = "info"  # The logging level of all loggers.

    @classmethod
    def from_cli(cls) -> Self:
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="cfg")
        return parser.parse_args().cfg

    @property
    def url(self):
        # Assumes we are using IPv4
        return f"http://{self.host}:{self.port}"
