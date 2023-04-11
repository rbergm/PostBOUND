"""Provides utilities to work with arbitrary collections like lists, sets and tuples."""
from __future__ import annotations

import itertools
import typing
from typing import Iterable, Sized, Iterator, Container, Collection

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


class Queue(Iterable[_T], Sized, Container[_T]):
    """Essentially, a queue is a wrapper around an underlying list of data that provides FIFO semantics."""

    def __init__(self, data: Iterable[_T] | None = None) -> None:
        self.data = list(data) if data else []

    def enqueue(self, value: _T) -> None:
        """Adds a new item to the end of the queue."""
        self.data.append(value)

    def append(self, value: _T) -> None:
        """Adds a new item at to end of the queue.

        Basically an alias for `enqueue` to enable easier interchangeability with normal lists.
        """
        self.enqueue(value)

    def extend(self, values: Iterable[_T]) -> None:
        """Adds all values to the end of the queue, in the order in which they are provided by the iterable."""
        self.data.extend(values)

    def head(self) -> _T | None:
        """Provides the current first element of the queue without removing."""
        return self.data[0] if self.data else None

    def pop(self) -> _T | None:
        """Provides the current first element of the queue and removes it."""
        item = self.head()
        if item:
            self.data.pop(0)
        return item

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, __x: object) -> bool:
        return __x in self.data

    def __iter__(self) -> Iterator[_T]:
        return self.data.__iter__()


class SizedQueue(Collection[_T]):
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
        """Adds a new item to the end of the queue, popping any excess items."""
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(value)

    def extend(self, values: typing.Iterable[_T]) -> None:
        """Adds all the items to the end of the queue, popping any excess items."""
        self.data = (self.data + list(values))[:self.capacity]

    def head(self) -> _T:
        """Provides the current first item of the queue without removing it."""
        return self.data[0]

    def pop(self) -> _T:
        """Provides the current first item of the queue and removes it."""
        return self.data.pop(0)

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
