"""Contains utilities to store and load objects more conveniently to/from JSON.

More specifically, this module introduces the `JsonizeEncoder`, which can be accessed via the `to_json` utility method.
This encoder allows to transform instances of any class to JSON by providing a `__json__` method in the class
implementation. This method does not take any (required) parameters and returns a JSON-izeable representation of the
current instance, e.g. a `dict` or a `list`.

Sadly (or luckily?), the inverse conversion does not work because JSON does not store any type information.
"""

from __future__ import annotations

import abc
import enum
import json
from pathlib import Path
from typing import IO, Any, Protocol, runtime_checkable

jsondict = dict
"""Type alias for a JSON-izeable dictionary."""


@runtime_checkable
class Jsonizable(Protocol):
    """Protocol to indicate that a certain class provides the `__json__` method."""

    @abc.abstractmethod
    def __json__(self) -> jsondict:
        raise NotImplementedError


class JsonizeEncoder(json.JSONEncoder):
    """The  JsonizeEncoder allows to transform instances of any class to JSON.

    This can be achieved by providing a `__json__` method in the class implementation. This method does not take any
    (required) parameters and returns a JSON-izeable representation of the current instance, e.g. a `dict` or a `list`.
    """

    def default(self, obj: Any) -> Any:
        if isinstance(obj, enum.Enum):
            return obj.value
        elif isinstance(obj, (set, frozenset)):
            return list(obj)
        elif isinstance(obj, Path):
            return str(obj)
        elif "__json__" in dir(obj):
            return obj.__json__()
        return json.JSONEncoder.default(self, obj)


def to_json(obj: Any, *args, **kwargs) -> str | None:
    """Utility to transform any object to a JSON object, while making use of the `JsonizeEncoder`.

    All arguments other than the object itself are passed to the default Python `json.dumps` function.
    """
    if obj is None:
        return None
    kwargs.pop("cls", None)
    return json.dumps(obj, *args, cls=JsonizeEncoder, **kwargs)


def to_json_dump(obj: Any, file: IO, *args, **kwargs) -> None:
    """Utility to transform any object to a JSON object and write it to a file, while making use of the `JsonizeEncoder`.

    All arguments other than the object itself are passed to the default Python `json.dump` function.
    """
    kwargs.pop("cls", None)
    json.dump(obj, file, cls=JsonizeEncoder, *args, **kwargs)
