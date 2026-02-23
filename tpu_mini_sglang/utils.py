import os
import sys
import traceback
from contextlib import suppress

import psutil


def get_exception_traceback():
    etype, value, tb = sys.exc_info()
    err_str = "".join(traceback.format_exception(etype, value, tb))
    return err_str


def kill_process_tree(include_parent: bool = True):
    """Kill all child process and optionally the process itself."""
    itself = psutil.Process(os.getpid())

    children = itself.children(recursive=True)
    for child in children:
        with suppress(psutil.NoSuchProcess):
            child.kill()

    if include_parent:
        itself.kill()
        sys.exit(0)
