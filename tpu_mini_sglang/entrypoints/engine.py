from tpu_mini_sglang.managers.tokenizer_manager import TokenizerManager
from tpu_mini_sglang.server_args import PortArgs, ServerArgs


def launch_subprocesses(server_args: ServerArgs):

    port_args = PortArgs.init_new()
    tokenizer_manager = TokenizerManager(server_args, port_args)

    return tokenizer_manager
