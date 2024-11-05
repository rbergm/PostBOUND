"""Contains various general errors that extend Python's base errors."""
from __future__ import annotations


class StateError(RuntimeError):
    """Indicates that an object is not in the right state to perform an operation.

    Parameters
    ----------
    *args
        Further details on which state was violated and how.
    """

    def __init__(self, *args) -> None:
        super().__init__(*args)


class InvariantViolationError(RuntimeError):
    """Indicates that some contract of a method was violated. The arguments should provide further details

    Parameters
    ----------
    *args
        Further details on what invariant was violated
    """

    def __init__(self, *args: object) -> None:
        super().__init__(*args)
