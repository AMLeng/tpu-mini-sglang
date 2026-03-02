import multiprocessing as mp

from tpu_mini_sglang.managers.detokenizer_manager import run_detokenizer_process
from tpu_mini_sglang.managers.tokenizer_manager import TokenizerManager
from tpu_mini_sglang.server_args import PortArgs, ServerArgs


def launch_subprocesses(server_args: ServerArgs) -> TokenizerManager:
    """
    Launch detokenizer_process as a separate process,
    and create a tokenizer_manager in the current process and return it.
    """

    port_args = PortArgs.init_new()

    detokenizer_process = mp.Process(target=run_detokenizer_process, args=(server_args, port_args))
    detokenizer_process.start()

    tokenizer_manager = TokenizerManager(server_args, port_args)

    return tokenizer_manager
