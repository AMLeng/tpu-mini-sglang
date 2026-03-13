import logging
import multiprocessing as mp
import signal

from tpu_mini_sglang.managers.detokenizer_manager import run_detokenizer_process
from tpu_mini_sglang.managers.scheduler import run_scheduler_process
from tpu_mini_sglang.managers.tokenizer_manager import TokenizerManager
from tpu_mini_sglang.server_args import PortArgs, ServerArgs
from tpu_mini_sglang.utils import kill_process_tree

logger = logging.getLogger(__name__)


def launch_subprocesses(server_args: ServerArgs) -> TokenizerManager:
    """
    Launch detokenizer_process as a separate process,
    and create a tokenizer_manager in the current process and return it.
    """

    def sigquit_handler(signum, frame):
        logger.error("Received sigquit from a child process. It usually means the child failed.")
        kill_process_tree()

    signal.signal(signal.SIGQUIT, sigquit_handler)

    # Use "spawn" instead of the default "fork".
    # This is necessary for JAX, where device state is initialized at import time.
    # Fork would duplicate that state, creating conflicts.
    mp.set_start_method("spawn", force=True)
    port_args = PortArgs.init_new()

    # Initialization pipe for blocking until the scheduler is ready
    scheduler_pipe_reader, scheduler_pipe_writer = mp.Pipe(duplex=False)

    # Since we work with JAX and TPUs, we only need a single scheduler process
    # in contrast to a GPU approach which needs one per GPU
    scheduler_process = mp.Process(
        target=run_scheduler_process, args=(server_args, port_args, scheduler_pipe_writer)
    )
    scheduler_process.start()

    detokenizer_process = mp.Process(target=run_detokenizer_process, args=(server_args, port_args))
    detokenizer_process.start()

    tokenizer_manager = TokenizerManager(server_args, port_args)

    # Block until scheduler is ready, then receive max_req_input_length data
    scheduler_data = scheduler_pipe_reader.recv()

    tokenizer_manager.max_req_input_length = scheduler_data["max_req_input_len"]

    return tokenizer_manager
