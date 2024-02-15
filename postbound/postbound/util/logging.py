"""Contains utilities to conveniently log different information."""
from __future__ import annotations

import atexit
import pprint
import functools
import sys
from collections.abc import Callable
from datetime import datetime
from typing import IO


def timestamp() -> str:
    """Provides the current time as a nice and normalized string."""
    return datetime.now().strftime("%y-%m-%d %H:%M:%S")


def make_logger(enabled: bool = True, *, file: IO[str] = sys.stderr, pretty: bool = False,
                prefix: str | Callable[[], str] = "") -> Callable:
    """Creates a new logging utility.

    The generated method can be used like a regular `print`, but with defaults that are better suited for logging purposes.

    If `enabled` is `False`, calling the logging function will not actually print anything and simply return. This
    is especially useful to implement logging-hooks in longer functions without permanently re-checking whether logging
    is enabled or not.

    By default, all logging output will be written to stderr, but this can be customized by supplying a different
    `file`.

    If `pretty` is enabled, structured objects such as dictionaries will be pretty-printed instead of being written
    on a single line. Note that pprint is used for all of the logging data everytime in that case.

    Parameters
    ----------
    enabled : bool, optional
        Whether logging is enabled, by default *True*
    file : IO[str], optional
        Destination to write the log entries to, by default ``sys.stderr``
    pretty : bool, optional
        Whether complex objects should be pretty-printed using the ``pprint`` module, by default *False*
    prefix : str | Callable[[], str], optional
        A common prefix that should be added before each log entry. Can be either a hard-coded string, or a callable that
        dynamically produces a string for each logging action separately (e.g. timestamp).

    Returns
    -------
    Callable
        _description_
    """
    def _log(*args, **kwargs) -> None:
        if prefix and isinstance(prefix, str):
            args = [prefix] + list(args)
        elif prefix:
            args = [prefix()] + list(args)
        print(*args, file=file, **kwargs)

    def _dummy_log(*args, **kwargs) -> None:
        pass

    if pretty and enabled:
        return functools.partial(pprint.pprint, stream=file)

    return _log if enabled else _dummy_log


def print_stderr(*args, **kwargs) -> None:
    """A normal `print` that writes to stderr instead of stdout."""
    kwargs.pop("file", None)
    print(*args, file=sys.stderr, **kwargs)


def print_if(should_print: bool, *args, use_stderr: bool = False, **kwargs) -> None:
    """A normal `print` that only prints something if `should_print` evaluates true-ish. Can optionally print to stderr."""
    if should_print:
        out_device = kwargs.get("file", sys.stderr if use_stderr else sys.stdout)
        print(*args, file=out_device, **kwargs)


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
