"""Provides utilities to work with arbitrary collections like lists, sets and tuples."""
from __future__ import annotations

import itertools
import typing
from collections.abc import Collection, Container, Generator, Iterator, Iterable, Sequence, Sized
from typing import Any, Optional

from postbound.util import dicts as dict_utils

T = typing.TypeVar("T")
"""The type parameter that is used for all generics in this module."""

ContainerType = typing.TypeVar("ContainerType", list, tuple, set, frozenset)
"""Specifies which types are considered containers.

For some methods this is necessary to determine whether any work still has to be done.
"""


def flatten(deep_list: Iterable[Iterable[T] | T]) -> list[T]:
    """Transforms a nested list into a flat list: ``[[1, 2], [3]]`` is turned into ``[1, 2, 3]``

    Parameters
    ----------
    deep_list : Iterable[Iterable[T]  |  T]
        The list to flatten

    Returns
    -------
    list[T]
        The flattened list: all elements of iterables from the `deep_list` are now contained directly in the resulting list.
    """
    flattened: list[T] = []
    for nested in deep_list:
        if isinstance(nested, Iterable) and not isinstance(nested, str):
            flattened.extend(nested)
        else:
            flattened.append(nested)
    return flattened


def enlist(obj: T | ContainerType[T], *, enlist_tuples: bool = False) -> ContainerType[T] | list[T]:
    """Transforms any object into a singular list of that object, if it is not a container already.

    Specifically, the following types are treated as container-like and will not be transformed: lists, tuples, sets
    and frozensets. The treatment of tuples can be configured via parameters. All other arguments will be wrapped in a list.

    For example, ``"abc"`` is turned into ``["abc"]``, whereas ``["abc"]`` is returned unmodified.

    Parameters
    ----------
    obj : T | ContainerType[T]
        The object or list to wrap
    enlist_tuples : bool, optional
        Whether a tuple `obj` should be enlisted. This is ``False`` by default

    Returns
    -------
    ContainerType[T] | list[T]
        The object, wrapped into a list if necessary
    """
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, tuple) and enlist_tuples:
        return [obj]
    list_types = [tuple, list, set, frozenset]
    if any(isinstance(obj, target_type) for target_type in list_types):
        return obj
    return [obj]


def get_any(elems: Iterable[T]) -> T:
    """Provides any element from an iterable. There is no guarantee which one will be returned.

    This method can potentially iterate over the entire iterable. The behaviour for empty iterables is undefined.

    Parameters
    ----------
    elems : Iterable[T]
        The items from which to choose.

    Returns
    -------
    T
        Any of the elements from the iterable. If the iterable is empty, the behaviour is undefined.
    """
    return next(elems)


def simplify(obj: Collection[T] | T) -> T | Collection[T]:
    """Unwraps containers containing just a single element.

    This can be thought of as the inverse operation to `enlist`. If the object contains multiple elements, nothing happens.

    Parameters
    ----------
    obj : Collection[T] | T
        The object to simplify

    Returns
    -------
    T | Collection[T]
        For a singular list, the object that was contained in that list. Otherwise `obj` is returned unmodified.

    Examples
    --------
    The singular list ``[1]`` is simplified to ``1``. On the other hand, ``[1,2]`` is returned unmodified.
    """
    if "__len__" not in dir(obj) or "__iter__" not in dir(obj):
        return obj

    if len(obj) == 1:
        return list(obj)[0]
    return obj


def powerset(lst: Collection[T]) -> Iterable[tuple[T, ...]]:
    """Calculates the powerset of the provided iterable.

    The powerset of a set *S* is defined as the set that contains all subsets of *S*. This is includes the empty set, as well
    as the entire set *S*.

    Parameters
    ----------
    lst : Collection[T]
        The "set" *S*

    Returns
    -------
    Iterable[tuple[T, ...]]
        The powerset of *S*. Each tuple correponds to a specific subset. The order of the elements within the tuple is not
        significant.
    """
    return itertools.chain.from_iterable(itertools.combinations(lst, size) for size in range(len(lst) + 1))


def sliding_window(lst: Sequence[T], size: int,
                   step: int = 1) -> Generator[tuple[Sequence[T], Sequence[T], Sequence[T]], None, None]:
    """Iterates over the given sequence using a sliding window.

    The window will contain exactly `size` many entries, starting at the beginning of the sequence. After yielding a
    window, the next window will be shifted `step` many elements.

    Parameters
    ----------
    lst : Sequence[T]
        The sequence to iterate over
    size : int
        The number of elements in the sliding window
    step : int, optional
        The number of elements to shift after each window, defaults to 1.

    Yields
    ------
    Generator[tuple[Sequence[T], Sequence[T], Sequence[T]]]
        The sliding window subsets. The tuples are structured as follows: *(prefix, window, suffix)* where *prefix* are all
        elements of the sequence before the current window, *window* contains exactly those elements that are part of the
        current window and *suffix* contains all elements after the current window.
    """
    for i in range(0, len(lst) - size + 1, step=step):
        prefix = lst[:i]
        window = lst[i:i + size]
        suffix = lst[i + size:]
        yield prefix, window, suffix


def pairs(lst: Iterable[T]) -> Generator[tuple[T, T], None, None]:
    """Provides all pairs of elements of the given iterable, disregarding order and identical pairs.

    This means that the resulting iterable will not contain entries *(a, a)* unless *a* itself is present multiple
    times in the input. Likewise, tuples *(a, b)* and *(b, a)* are treated as equal and only one of them will be
    returned (Again, unless *a* or *b* are present multiple times in the input. In that case, their order is
    unspecified.)

    Parameters
    ----------
    lst : Iterable[T]
        The iterable that contains the pairs. It must be possible to iterate over it multiple times (twice, to be exact).

    Yields
    ------
    Generator[tuple[T, T], None, None]
        The element pairs.
    """
    for a_idx, a in enumerate(lst):
        for b_idx, b in enumerate(lst):
            if b_idx <= a_idx:
                continue
            yield a, b


def set_union(sets: Iterable[set[T] | frozenset[T]]) -> set[T]:
    """Computes the union of many sets.

    Parameters
    ----------
    sets : Iterable[set[T]  |  frozenset[T]]
        The sets to combine. Frozensets are "expanded" to regular sets.

    Returns
    -------
    set[T]
        Large union of all provided sets.
    """
    union_set: set[T] = set()
    for s in sets:
        union_set |= s
    return union_set


def make_hashable(obj: Any) -> Any:
    """Attempts to generate an equivalent, hashable representation for a container.

    This function operates on the standard container types list, tuple, set, dictionary and frozenset and performs the
    following conversion:

    - list becomes tuple, all elements of the list are recursively made hashable
    - tuples are left as-is, but all elements of the tuple are recursively made hashable
    - sets become frozensets. The elements are left as they are, because they must already be hashable
    - dictionaries become instances of `dict_utils.HashableDict`. The values are recursively made hashable, keys are left the
      way they are because they must already be hashable
    - frozensets are left as-is

    All other types, including user-defined types are returned as-is.

    Parameters
    ----------
    obj : Any
        The object to hash

    Returns
    -------
    Any
        The hashable counterpart of the object
    """
    if isinstance(obj, set):
        return frozenset(obj)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return tuple(make_hashable(elem) for elem in obj)
    elif isinstance(obj, dict):
        return dict_utils.HashableDict({k: make_hashable(v) for k, v in obj.items()})
    else:
        return obj


class Queue(Iterable[T], Sized, Container[T]):
    """A queue is a wrapper around an underlying list of elements which provides FIFO semantics for access.

    Parameters
    ----------
    data : Iterable[T] | None, optional
        Initial contents of the queue. By default the queue is empty at the beginning.

    """

    def __init__(self, data: Iterable[T] | None = None) -> None:
        self.data = list(data) if data else []

    def enqueue(self, value: T) -> None:
        """Adds a new item to the end of the queue.

        Parameters
        ----------
        value : T
            The item to add
        """
        self.data.append(value)

    def push(self, value: T) -> None:
        """Adds a new item to the end of the queue.

        This is an alias for `enqueue`.

        Parameters
        ----------
        value : T
            The item to add
        """
        self.enqueue(value)

    def append(self, value: T) -> None:
        """Adds a new item to end of the queue.

        This method is an alias for `enqueue` to enable easier interchangeability with normal lists.

        Parameters
        ----------
        value : T
            The item to add
        """
        self.enqueue(value)

    def extend(self, values: Iterable[T]) -> None:
        """Adds a number of values to the end of the queue.

        Parameters
        ----------
        values : Iterable[T]
            The elements to add. The order in the queue matches the order in the iterable.
        """
        self.data.extend(values)

    def head(self) -> Optional[T]:
        """Provides the current first element of the queue without removing.

        Returns
        -------
        Optional[T]
            The first element if it exists, or ``None`` if the queue is empty.
        """
        return self.data[0] if self.data else None

    def peak(self) -> Optional[T]:
        """Provides the current first element of the queue without removing.

        This is an alias for `head`.

        Returns
        -------
        Optional[T]
            The first element if it exists, or ``None`` if the queue is empty.
        """
        return self.head()

    def pop(self) -> Optional[T]:
        """Provides the current first element of the queue and removes it.

        Returns
        -------
        Optional[T]
            The first element if it exists, or ``None`` if the queue is empty.
        """
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

    def __repr__(self) -> str:
        return f"Queue({self.data})"

    def __str__(self) -> str:
        return str(self.data)


class SizedQueue(Collection[T]):
    """A sized queue extends on the behaviour of a normal queue by restricting the number of items in the queue.

    A sized queue has weak FIFO semantics: items can only be appended at the end, but the contents of the entire queue
    can be accessed at any time.

    If upon enqueuing a new item the queue is already at maximum capacity, the current head of the queue will be
    dropped.

    Parameters
    ----------
    capacity : int
        The maximum number of items the queue can contain at the same time.
    data : Optional[Iterable[T]], optional
        Initial contents of the queue. By default the queue is empty at the beginning.

    Notes
    -----
    Although `Queue` and `SizedQueue` provide similar FIFO semantics, there is no subclass relationship between the two. This
    is by design, since the contract of a queue is very different from the contract of a sized queue.

    """

    def __init__(self, capacity: int, data: Optional[Iterable[T]] = None) -> None:
        self.data = list(data) if data else []
        self.capacity = capacity

    def append(self, value: T) -> None:
        """Adds a new item to the end of the queue, popping any excess items.

        Parameters
        ----------
        value : T
            The value to add
        """
        if len(self.data) >= self.capacity:
            self.data.pop(0)
        self.data.append(value)

    def extend(self, values: typing.Iterable[T]) -> None:
        """Adds all the items to the end of the queue, popping any excess items.

        Parameters
        ----------
        values : typing.Iterable[T]
            The values to add
        """
        self.data = (self.data + list(values))[:self.capacity]

    def head(self) -> Optional[T]:
        """Provides the current first item of the queue without removing it.

        Returns
        -------
        Optional[T]
            The first item in the queue, or ``None`` if the queue is empty
        """
        return self.data[0] if self.data else None

    def pop(self) -> Optional[T]:
        """Provides the current first item of the queue and removes it.

        Returns
        -------
        Optional[T]
            The first item in the queue, or ``None`` if the queue is empty
        """
        return self.data.pop(0) if self.data else None

    def __contains__(self, other: T) -> bool:
        return other in self.data

    def __iter__(self) -> typing.Iterator[T]:
        return self.data.__iter__()

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"SizedQueue(capacity={self.capacity}, data={self.data})"

    def __str__(self) -> str:
        return str(self.data)
