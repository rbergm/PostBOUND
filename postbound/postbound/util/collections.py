"""Provides utilities to work with arbitrary collections like lists, sets and tuples."""
from __future__ import annotations

import itertools
import typing
from collections.abc import Collection, Container, Generator, Iterator, Iterable, Sequence, Sized

T = typing.TypeVar("T")
ContainerType = typing.TypeVar("ContainerType", list, tuple, set, frozenset)


def flatten(deep_list: Iterable[Iterable[T]]) -> list[T]:
    """Transforms a nested list into a flat list: `[[1, 2], [3]]` is turned into `[1, 2, 3]`"""
    return list(itertools.chain(*deep_list))


def enlist(obj: T | ContainerType[T], *, enlist_tuples: bool = False) -> ContainerType[T] | list[T]:
    """Transforms any object into a singular list, if it is not a container already.

    Specifically, the following types are treated as container-like and will not be transformed: lists, tuples, sets
    and frozensets. All other arguments will be wrapped in a list. If the argument is a tuple and enlist_tuples is
    `True`, it will also be wrapped in a list.

    For example, `"abc"` is turned into `["abc"]`, whereas `["abc"]` is returned unmodified.
    """
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, tuple) and enlist_tuples:
        return [obj]
    list_types = [tuple, list, set, frozenset]
    if any(isinstance(obj, target_type) for target_type in list_types):
        return obj
    return [obj]


def simplify(obj: Collection[T]) -> T | Iterable[T]:
    """Unwraps singular containers.

    For example `[1]` is simplified to `1`. On the other hand, `[1,2]` is returned unmodified.
    """
    if len(obj) == 1:
        return list(obj)[0]
    return obj


def powerset(lst: Collection[T]) -> Iterable[tuple[T, ...]]:
    """Calculates the powerset of the provided iterable."""
    return itertools.chain.from_iterable(itertools.combinations(lst, size) for size in range(len(lst) + 1))


def sliding_window(lst: Sequence[T], size: int,
                   step: int = 1) -> Generator[tuple[Sequence[T], Sequence[T], Sequence[T]]]:
    """Iterates over the given sequence using a sliding window.

    The window will contain exactly `size` many entries, starting at the beginning of the sequence. After yielding a
    window, the next window will be shifted `step` many elements.

    The tuples produced by the generator are structured as follows: `(prefix, window, suffix)` where `prefix` are all
    elements of the sequence before the current window, `window` contains exactly those elements that are part of the
    current window and `suffix` contains all elements after the current window.
    """
    for i in range(0, len(lst) - size + 1, step=step):
        prefix = lst[:i]
        window = lst[i:i + size]
        suffix = lst[i + size:]
        yield prefix, window, suffix


def pairs(lst: Iterable[T]) -> Iterable[tuple[T, T]]:
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


class Queue(Iterable[T], Sized, Container[T]):
    """Essentially, a queue is a wrapper around an underlying list of data that provides FIFO semantics."""

    def __init__(self, data: Iterable[T] | None = None) -> None:
        self.data = list(data) if data else []

    def enqueue(self, value: T) -> None:
        """Adds a new item to the end of the queue."""
        self.data.append(value)

    def append(self, value: T) -> None:
        """Adds a new item at to end of the queue.

        Basically an alias for `enqueue` to enable easier interchangeability with normal lists.
        """
        self.enqueue(value)

    def extend(self, values: Iterable[T]) -> None:
        """Adds all values to the end of the queue, in the order in which they are provided by the iterable."""
        self.data.extend(values)

    def head(self) -> T | None:
        """Provides the current first element of the queue without removing."""
        return self.data[0] if self.data else None

    def pop(self) -> T | None:
        """Provides the current first element of the queue and removes it."""
        item = self.head()
        if item:
            self.data.pop(0)
        return item

    def __len__(self) -> int:
        return len(self.data)

    def __contains__(self, __x: object) -> bool:
        return __x in self.data

    def __iter__(self) -> Iterator[T]:
        return self.data.__iter__()


class SizedQueue(Collection[T]):
    """A sized queue extends on the behaviour of a normal queue by restricting the number of items in the queue.

    A sized queue has weak FIFO semantics: items can only be appended at the end, but the contents of the entire queue
    can be accessed at any time.
    If upon enqueuing a new item the queue is already at maximum capacity, the current head of the queue will be
    dropped.
    """

    def __init__(self, capacity: int) -> None:
        self.data = []
        self.capacity = capacity

    def append(self, value: T) -> None:
        """Adds a new item to the end of the queue, popping any excess items."""
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(value)

    def extend(self, values: typing.Iterable[T]) -> None:
        """Adds all the items to the end of the queue, popping any excess items."""
        self.data = (self.data + list(values))[:self.capacity]

    def head(self) -> T:
        """Provides the current first item of the queue without removing it."""
        return self.data[0]

    def pop(self) -> T:
        """Provides the current first item of the queue and removes it."""
        return self.data.pop(0)

    def __contains__(self, other: T) -> bool:
        return other in self.data

    def __iter__(self) -> typing.Iterator[T]:
        return self.data.__iter__()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return str(self.data)
