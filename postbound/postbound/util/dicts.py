"""Contains utilties to access and modify dictionaries more conveniently."""

import typing
from typing import Dict

_K = typing.TypeVar("_K")
_V = typing.TypeVar("_V")


def key(dictionary: Dict[_K, _V]) -> _K:
    """Provides the key of a dictionary with just 1 item."""
    if not len(dictionary) == 1:
        raise ValueError("Dictionary must contain exactly 1 entry, not " + len(dictionary))
    for key in dictionary:
        return key
