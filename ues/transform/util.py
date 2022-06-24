
import itertools
import numbers
import typing
from typing import List, Dict, Sized, Union, Callable


_T = typing.TypeVar("_T")
_K = typing.TypeVar("_K")
_V = typing.TypeVar("_V")


def head(lst: List[_T]) -> _T:
    """Provides the first element of a list. Raises `ValueError` if list is empty."""
    if not len(lst):
        raise ValueError("List is empty")
    return lst[0]


def dict_key(dictionary: Dict[_K, _V], *, pull_any: bool = False) -> _K:
    """For a dictionary with just one entry, provides the key of that entry.

    If multiple entries exist and pull_any is `True`, provides any of the keys. Otherwise raises a ValueError.
    """
    if not isinstance(dictionary, dict):
        raise TypeError("Not a dict: " + str(dictionary))
    if not dictionary:
        raise ValueError("No entries")
    keys = list(dictionary.keys())
    if len(keys) > 1 and not pull_any:
        raise ValueError("Ambigous call - dict contains multiple entries: " + str(dictionary))
    return next(iter(keys))


def dict_value(dictionary: Dict[_K, _V], *, pull_any: bool = False) -> _V:
    """For a dictionary with just one entry, provides the value of that entry.

    If multiple entries exist and pull_any is `True`, provides any of the values. Otherwise raises a ValueError.
    """
    if not isinstance(dictionary, dict):
        raise TypeError("Not a dict: " + str(dictionary))
    if not dictionary:
        raise ValueError("No entries")
    vals = list(dictionary.values())
    if len(vals) > 1 and not pull_any:
        raise ValueError("Ambigous call - dict contains multiple entries: " + str(dictionary))
    return next(iter(vals))


def dict_merge(a: Dict[_K, _V], b: Dict[_K, _V], *, update: Callable[[_K, _V, _V], _V] = None) -> Dict[_K, _V]:
    """Creates a new dict containing all key/values pairs from both argument dictionaries.

    If keys overlap, entries from dictionary `b` will take priority, unless an `update` method is given.
    If `update` is given, and `a[k] = v` and `b[k] = v'` (i.e. both `a` and `b` share a key `k`) the merged dictionary
    will contain the result of `update(k, v, v')` as entry for `k`.

    Note that as of Python 3.9, such a method was added to dictionaries as well (via the `|=` syntax).
    """
    if not update:
        return dict([*a.items()] + [*b.items()])
    else:
        merged = dict(a)
        for key, val in b.items():
            if key in merged:
                merged[key] = update(key, merged[key], val)
            else:
                merged[key] = val
        return merged


def flatten(deep_lst: List[Union[List[_T], _T]], *, recursive=False) -> List[_T]:
    """Unwraps all nested lists, leaving scalar values untouched.

    E.g. for a deep list `[[1, 2, 3], 4, [5, 6]]` will return `[1, 2, 3, 4, 5, 6]` (mind the scalar 4).

    If `recursive` is `True`, this process will continue until all nested lists are flattened (e.g. in the case of
    `[[[1,2]]]`).
    """
    deep_lst = [[deep_elem] if not isinstance(deep_elem, list) else deep_elem for deep_elem in deep_lst]
    flattened = list(itertools.chain(*deep_lst))
    if recursive and any(isinstance(deep_elem, list) for deep_elem in flattened):
        return flatten(flattened, recursive=True)
    return flattened


def enlist(obj: _T, strict: bool = True) -> List[_T]:
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
    """Unwraps single scalar values from list-like (i.e. list or tuple) containers.

    E.g. `simplify([42])` will return 42, whereas `simplify([24, 42])` will return `[24, 42]`.
    """
    while (isinstance(lst, list) or isinstance(lst, tuple)) and len(lst) == 1:
        lst = lst[0]
    return lst


def represents_number(val: str) -> bool:
    """Checks if `val` can be cast into an integer/float value."""
    try:
        float(val)
    except (TypeError, ValueError):
        return False
    return True


def argmin(mapping: Dict[_K, numbers.Number]) -> _K:
    """
    For a dict mapping keys to numeric types, returns the key `k` with minimum value `v`, s.t. for all keys `k'` with
    values `v'` it holds that `v <= v'`.
    """
    return min(mapping, key=mapping.get)


def contains_multiple(obj: Sized):
    """Checks whether an object is list-like (i.e. has a length) and contains more than 1 entries."""
    if "__len__" not in dir(obj):
        return False
    return len(obj) > 1


def _log(*args, **kwargs):
    print(*args, **kwargs)


def _dummy_log(*args, **kwargs):
    pass


def make_logger(enabled: bool = True):
    return _log if enabled else _dummy_log
