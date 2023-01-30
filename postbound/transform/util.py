
import itertools
import typing
from typing import List, Set, Any, Iterable, Union


_T = typing.TypeVar("_T")


def head(lst: List[_T]) -> _T:
    """Provides the first element of a list. Raises `ValueError` if list is empty."""
    if not len(lst):
        raise ValueError("List is empty")
    return lst[0]


def flatten(deep_lst: List[Union[List[_T], _T]], *, recursive: bool = False, flatten_set: bool = False) -> List[_T]:
    """Unwraps all nested lists, leaving scalar values untouched.

    E.g. for a deep list `[[1, 2, 3], 4, [5, 6]]` will return `[1, 2, 3, 4, 5, 6]` (mind the scalar 4).

    If `recursive` is `True`, this process will continue until all nested lists are flattened (e.g. in the case of
    `[[[1,2]]]`).

    If `flatten_set` is `True`, sets will be flattened just the same as lists will.
    """
    def check_flattenable(elem, flatten_sets: bool = False):
        return isinstance(elem, list) or (flatten_sets and isinstance(elem, set))

    deep_lst = [[deep_elem] if not check_flattenable(deep_elem, flatten_set) else deep_elem for deep_elem in deep_lst]
    flattened = list(itertools.chain(*deep_lst))
    if recursive and any(check_flattenable(deep_elem, flatten_set) for deep_elem in flattened):
        return flatten(flattened, recursive=True)
    return flattened


def enlist(obj: _T, *, strict: bool = True) -> List[_T]:
    """Turns a scalar value into a list, if it is not one already.

    E.g. `enlist(42)` will return `[42]`, whereas `enlist([24])` returns `[24]`.

    Setting `strict` to `True` (the default) will always enlist, except for `list` arguments (e.g. tuples will also be
    wrapped). Setting `strict` to `False` will only enlist objects that are not iterable.
    """
    if strict:
        return obj if isinstance(obj, list) else [obj]
    else:
        return obj if "__iter__" in dir(obj) else [obj]


def simplify(lst: List[_T]) -> Union[_T, List[_T]]:
    """Unwraps single scalar values from list-like (i.e. list or tuple or set) containers.

    E.g. `simplify([42])` will return 42, whereas `simplify([24, 42])` will return `[24, 42]`.
    """
    while (isinstance(lst, list) or isinstance(lst, tuple) or isinstance(lst, set)) and len(lst) == 1:
        lst = list(lst)[0]
    return lst


def powerset(lst: Iterable[_T]) -> Iterable[Set[_T]]:
    """Calculates the powerset of the provided iterable."""
    return itertools.chain.from_iterable(itertools.combinations(lst, size) for size in range(len(lst) + 1))


def contains_multiple(obj: Any):
    """Checks whether an object is list-like (i.e. has a length) and contains more than 1 entries.

    Note that strings and tuples are treated as scalar objects.
    """
    if "__len__" not in dir(obj) or isinstance(obj, str) or isinstance(obj, tuple):
        return False
    return len(obj) > 1


def pull_any(iterable: Iterable[_T], *, strict: bool = True) -> _T:
    """Retrieves any element from the iterable.

    If `strict` is `False` and a scalar object is provided, this object will be returned as-is. In addition, in
    non-strict mode, `None` will be returned for empty iterables.
    """
    if not strict and not contains_multiple(iterable):
        return iterable
    if strict and not iterable:
        raise ValueError("Empty iterable")
    return next(iter(iterable), None)


class Version:
    def __init__(self, ver: Union[str, int, List[str], List[int]]):
        if isinstance(ver, int):
            self._version = [ver]
        elif isinstance(ver, str):
            self._version = [int(v) for v in ver.split(".")]
        elif isinstance(ver, list) and ver:
            self._version = [int(v) for v in ver]
        else:
            raise ValueError(f"Unknown version string: {ver}")

    def _wrap_version(self, other) -> "Version":
        return other if isinstance(other, Version) else Version(other)

    def __eq__(self, __o: object) -> bool:
        try:
            other = self._wrap_version(__o)
            if not len(self) == len(other):
                return False
            for i in range(len(self)):
                if not self._version[i] == other._version[i]:
                    return False
            return True
        except ValueError:
            return False

    def __ge__(self, __o: object) -> bool:
        return not self < __o

    def __gt__(self, __o: object) -> bool:
        return not self <= __o

    def __le__(self, __o: object) -> bool:
        other = self._wrap_version(__o)
        for comp in zip(self._version, other._version):
            own_version, other_version = comp
            if own_version < other_version:
                return True
            if other_version < own_version:
                return False
        return len(self) <= len(other)

    def __lt__(self, __o: object) -> bool:
        other = self._wrap_version(__o)
        for comp in zip(self._version, other._version):
            own_version, other_version = comp
            if own_version < other_version:
                return True
            if other_version < own_version:
                return False
        return len(self) < len(other)

    def __len__(self) -> int:
        return len(self._version)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return ".".join(str(v) for v in self._version)
