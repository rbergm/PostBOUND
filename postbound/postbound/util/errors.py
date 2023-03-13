"""Contains various errors that extend the base errors from Python and are not specific to certain parts of PostBOUND."""
from __future__ import annotations


class StateError(RuntimeError):
    """Indicates that an object is not in the right state to perform an operation."""

    def __init__(self, msg: str = "") -> None:
        super().__init__(msg)
