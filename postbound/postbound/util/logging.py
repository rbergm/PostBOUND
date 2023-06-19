"""Contains utilities to conveniently log different information."""
from __future__ import annotations

import atexit
import pprint
import functools
import sys
from datetime import datetime
from typing import Callable, IO


def timestamp() -> str:
    """Provides the current time as a nice and normalized string."""
    return datetime.now().strftime("%y-%m-%d %H:%M:%S")


def make_logger(enabled: bool = True, *, file: IO[str] = sys.stderr, pretty: bool = False) -> Callable:
    """Creates a new logging utility.

    The generated method can be used like a regular `print`, but with better defaults.

    If `enabled` is `False`, calling the logging function will not actually print anything and simply return. This
    is especially useful to implement logging-hooks in longer functions without permanently re-checking whether logging
    is enabled or not.

    By default, all logging output will be written to stdout, but this can be customized by supplying a different
    `file`.

    If `pretty` is enabled, structured objects such as dictionaries will be pretty-printed instead of being written
    on a single line. Note that pprint is used for all of the logging data everytime in that case.
    """

    def _log(*args, **kwargs):
        print(*args, file=file, **kwargs)

    def _dummy_log(*args, **kwargs):
        pass

    if pretty and enabled:
        return functools.partial(pprint.pprint, stream=file)

    return _log if enabled else _dummy_log


def print_stderr(*args, **kwargs) -> None:
    """A normal `print` that writes to stderr instead of stdout."""
    kwargs.pop("file", None)
    print(*args, file=sys.stderr, **kwargs)


def print_if(should_print: bool, *args, **kwargs) -> None:
    """A normal `print` that only prints something if `should_print` evaluates true-ish."""
    if should_print:
        print(*args, **kwargs)


class _TeeLogger:
    def __init__(self, target_file: str, output_mode: str = "a") -> None:
        self._original_stdout = sys.stdout
        self._log_out = open(target_file, output_mode)
        atexit.register(lambda: self._log_out.close())

    def write(self, message: str) -> None:
        self._original_stdout.write(message)
        self._log_out.write(message)


def tee_stdout(target_file: str, output_mode: str = "a") -> None:
    sys.stdout = _TeeLogger(target_file, output_mode)
