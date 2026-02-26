from tpu_mini_sglang.managers.tokenizer_manager import TokenizerManager
from tpu_mini_sglang.server_args import ServerArgs


def launch_subprocesses(server_args: ServerArgs):

    tokenizer_manager = TokenizerManager(server_args)

    return tokenizer_manager
