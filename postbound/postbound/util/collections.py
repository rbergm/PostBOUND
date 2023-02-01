import itertools
import typing
from typing import Iterable, Sized

_T = typing.TypeVar("_T")


def powerset(lst: (Iterable[_T], Sized)) -> Iterable[tuple[_T, ...]]:
    """Calculates the powerset of the provided iterable."""
    return itertools.chain.from_iterable(itertools.combinations(lst, size) for size in range(len(lst) + 1))


def pairs(lst: Iterable[_T]) -> Iterable[tuple[_T, _T]]:
    all_pairs = []
    for a_idx, a in enumerate(lst):
        for b_idx, b in enumerate(lst):
            if b_idx <= a_idx:
                continue
            all_pairs.append((a, b))
    return all_pairs


class SizedQueue(typing.Iterable[_T]):
    """A sized queue extends on the behaviour of a normal queue by restricting the number of items in the queue.

    A sized queue has weak FIFO semantics: items can only be appended at the end, but the contents of the entire queue can be
    accessed at any time.
    If upon enqueuing a new item the queue is already at maximum capacity, the current head of the queue will be dropped.
    """

    def __init__(self, capacity: int) -> None:
        self.data = []
        self.capacity = capacity

    def append(self, value: _T) -> None:
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(value)

    def extend(self, values: typing.Iterable[_T]) -> None:
        self.data = (self.data + values)[:self.capacity]

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
