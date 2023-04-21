"""Provides additional type hints, type decorators, ..."""
from __future__ import annotations

import functools
import warnings
from typing import Callable


def deprecated(func: Callable) -> Callable:
    """Indicates that the given function or class should no longer be used."""

    @functools.wraps(func)
    def deprecation_wrapper(*args, **kwargs) -> Callable:
        warnings.warn(f"Usage of {func.__name__} is deprecated")
        return func(*args, **kwargs)

    return deprecation_wrapper
