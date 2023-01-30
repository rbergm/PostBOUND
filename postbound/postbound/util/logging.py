
import pprint
import functools
import sys
from datetime import datetime
from typing import IO


def timestamp() -> str:
    return datetime.now().strftime("%y-%m-%d %H:%M:%S")


def make_logger(enabled: bool = True, *, file: IO[str] = sys.stderr, pretty: bool = False):
    def _log(*args, **kwargs):
        print(*args, file=file, **kwargs)

    def _dummy_log(*args, **kwargs):
        pass

    if pretty and enabled:
        return functools.partial(pprint.pprint, stream=file)

    return _log if enabled else _dummy_log


def print_stderr(*args, **kwargs):
    """Prints to stderr rather than stdout."""
    kwargs.pop("file", None)
    print(*args, file=sys.stderr, **kwargs)


def print_if(should_print: bool, *args, **kwargs):
    """Prints, only if the first argument is True-ish."""
    if should_print:
        print(*args, **kwargs)
