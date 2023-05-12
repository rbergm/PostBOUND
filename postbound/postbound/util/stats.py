"""Different mathematical and statistical formulas and utilties."""
from __future__ import annotations

import math


def catalan_number(n: int) -> int:
    """Computes the n-th catalan number. See https://en.wikipedia.org/wiki/Catalan_number."""
    return round(math.comb(2 * n, n) / (n + 1))


def jaccard(a: set | frozenset, b: set | frozenset) -> float:
    """Jaccard coefficient between a and b. Defined as |a ∩ b| / |a ∪ b|"""
    return len(a & b) / len(a | b)
