"""Provides utilities to work with arbitrary collections like lists, sets and tuples."""
from __future__ import annotations

import itertools
import typing
from typing import Iterable, Sized

_T = typing.TypeVar("_T")


def flatten(deep_list: Iterable[Iterable[_T]]) -> list[_T]:
    """Transforms a nested list into a flat list: `[[1, 2], [3]]` is turned into `[1, 2, 3]`"""
    return list(itertools.chain(*deep_list))


def enlist(obj: _T | list[_T]) -> list[_T]:
    """Transforms any object into a singular list, if it is not a list already.

    For example, `"abc"` is turned into `["abc"]`, whereas `["abc"]` is returned unmodified.
    """
    if isinstance(obj, list):
        return obj
    return [obj]


def simplify(obj: (Iterable[_T], Sized)) -> _T | Iterable[_T]:
    """Unwraps singular containers.

    For example `[1]` is simplified to `1`. On the other hand, `[1,2]` is returned unmodified.
    """
    if len(obj) == 1:
        return list(obj)[0]
    return obj


def powerset(lst: (Iterable[_T], Sized)) -> Iterable[tuple[_T, ...]]:
    """Calculates the powerset of the provided iterable."""
    return itertools.chain.from_iterable(itertools.combinations(lst, size) for size in range(len(lst) + 1))


def pairs(lst: Iterable[_T]) -> Iterable[tuple[_T, _T]]:
    """Provides all pairs of elements of the given iterable, disregarding order and identical pairs.

    This means that the resulting iterable will not contain entries `(a, a)` unless `a` itself is present multiple
    times in the input. Likewise, tuples `(a, b)` and `(b, a)` are treated equally and only one of them will be
    returned (Again, unless `a` or `b` are present multiple times in the input. In that case, their order is
    unspecified.)
    """
    all_pairs = []
    for a_idx, a in enumerate(lst):
        for b_idx, b in enumerate(lst):
            if b_idx <= a_idx:
                continue
            all_pairs.append((a, b))
    return all_pairs


def set_union(sets: Iterable[set]) -> set:
    """Combines the elements of all input sets into one large set."""
    union_set = set()
    for s in sets:
        union_set |= s
    return union_set


class SizedQueue(typing.Iterable[_T]):
    """A sized queue extends on the behaviour of a normal queue by restricting the number of items in the queue.

    A sized queue has weak FIFO semantics: items can only be appended at the end, but the contents of the entire queue
    can be accessed at any time.
    If upon enqueuing a new item the queue is already at maximum capacity, the current head of the queue will be
    dropped.
    """

    def __init__(self, capacity: int) -> None:
        self.data = []
        self.capacity = capacity

    def append(self, value: _T) -> None:
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(value)

    def extend(self, values: typing.Iterable[_T]) -> None:
        self.data = (self.data + list(values))[:self.capacity]

    def head(self) -> _T:
        return self.data[0]

    def __contains__(self, other: _T) -> bool:
        return other in self.data

    def __iter__(self) -> typing.Iterator[_T]:
        return self.data.__iter__()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.data)
