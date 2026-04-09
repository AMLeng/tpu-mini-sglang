"""Launch the inference server."""

from tpu_mini_sglang.entrypoints.http_server import launch_server
from tpu_mini_sglang.server_args import ServerArgs
from tpu_mini_sglang.utils import kill_process_tree

if __name__ == "__main__":
    server_args = ServerArgs.build_parser(dest="cfg").parse_args().cfg
    try:
        launch_server(server_args)
    finally:
        kill_process_tree(include_parent=False)
