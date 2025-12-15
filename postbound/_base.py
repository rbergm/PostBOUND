from __future__ import annotations

from pathlib import Path
from typing import TypeVar

T = TypeVar("T")
"""Typed expressions use this generic type variable."""

pbdir = Path.home() / ".postbound"
