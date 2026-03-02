import asyncio
import os
import sys
import traceback
from contextlib import suppress

import psutil
import zmq


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


def get_zmq_socket(context: zmq.Context, socket_type: int, endpoint: str, bind: bool) -> zmq.Socket:
    # zmq.SocketType is an int enum type but e.g. zmq.PUSH is an int
    # We annotate socket_type with int so that mypy doesn't complain

    socket = context.socket(socket_type)

    # High Water Mark settings allow for unlimited queue
    # BUFfer settings use OS default socket buffer size
    if socket_type == zmq.PUSH:
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.SNDBUF, -1)
    elif socket_type == zmq.PULL:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.RCVBUF, -1)
    else:
        raise ValueError(f"Unsupported socket type {socket_type}")

    if bind:
        socket.bind(endpoint)
    else:
        socket.connect(endpoint)
    return socket


def get_or_create_event_loop() -> asyncio.AbstractEventLoop:
    """Gets the running event loop or creates a new one if it doesn't exist."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
