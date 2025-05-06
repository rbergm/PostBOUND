"""Contains various general errors that extend Python's base errors."""
from __future__ import annotations


class LogicError(RuntimeError):
    """Generic error to indicate that any kind of algorithmic problem occurred.

    This error is generally used when some assumption within PostBOUND is violated, but it's (probably) not the user's fault.
    As a rule of thumb, if the user supplies faulty input, a `ValueError` should be raised instead.
    Therefore, encoutering a `LogicError` indicates a bug in PostBOUND itself.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class StateError(RuntimeError):
    """Indicates that an object is not in the right state to perform an operation."""
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)


class InvariantViolationError(LogicError):
    """Indicates that some contract of a method was violated. The arguments should provide further details."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
