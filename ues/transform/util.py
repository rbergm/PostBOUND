
import argparse
import collections
import collections.abc
import itertools
import numbers
import os
import sys
import typing
import warnings
from typing import List, Dict, Any, Iterable, Tuple, Union, Callable, IO

import psycopg2

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
        raise TypeError("Not a dict: " + str(dictionary) + f" ({type(dictionary)})")
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
        raise TypeError("Not a dict: " + str(dictionary) + f" ({type(dictionary)})")
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


def dict_update(dictionary: Dict[_K, _V], update: Callable[[_K, _V], _T]) -> Dict[_K, _T]:
    """Creates a new dict by calling update on each key/value pair on the old dict, retaining its keys."""
    return {key: update(val) for key, val in dictionary.items()}


def dict_explode(dictionary: Dict[_K, List[_V]]) -> List[Tuple[_K, _V]]:
    """Transforms dicts mapping keys to lists of values to a list of key/value pairs."""
    values = []
    for key, dict_values in dictionary.items():
        values.extend(zip(itertools.cycle([key]), dict_values))
    return values


def dict_hash(dictionary: Dict[_K, _V]) -> int:
    """Calculates a hash value based on the current dict contents (keys and values)."""
    keys = list(dictionary.keys())
    values = []
    for val in dictionary.values():
        if isinstance(val, collections.abc.Hashable):
            values.append(hash(val))
        elif isinstance(val, list) or isinstance(val, set):
            values.append(hash(tuple(val)))
        elif isinstance(val, dict):
            values.append(dict_hash(val))
        else:
            warnings.warn("Unhashable type: " + type(val))
    keys_hash = hash(tuple(keys))
    values_hash = hash(tuple(values))
    return hash((keys_hash, values_hash))


def dict_generate_multi(entries: List[Tuple[_K, _V]]) -> Dict[_K, List[_V]]:
    """Generates a dict based on its entries.

    Each key can occur multiple times and values will be aggregated in a list.
    """
    collector = collections.defaultdict(list)
    for key, value in entries:
        collector[key].append(value)
    return dict(collector)


def dict_reduce_multi(multi_dict: Dict[_K, List[_V]], reduction: Callable[[_K, List[_V]], _V]) -> Dict[_K, _V]:
    """De-groups a multi-dict by aggregating the values based on key and values."""
    return {key: reduction(key, values) for key, values in multi_dict.items()}


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


def make_logger(enabled: bool = True, *, file: IO[str] = sys.stderr):
    def _log(*args, **kwargs):
        print(*args, file=file, **kwargs)

    def _dummy_log(*args, **kwargs):
        pass

    return _log if enabled else _dummy_log


def connect_postgres(parser: argparse.ArgumentParser, conn_str: str = None):
    if not conn_str:
        if not os.path.exists(".psycopg_connection"):
            parser.error("No connect string for psycopg given.")
        with open(".psycopg_connection", "r") as conn_file:
            conn_str = conn_file.readline().strip()
    conn = psycopg2.connect(conn_str)
    return conn


def print_stderr(*args, **kwargs):
    """Prints to stderr rather than stdout."""
    kwargs.pop("file", None)
    print(*args, file=sys.stderr, **kwargs)


def print_if(should_print: bool, *args, **kwargs):
    """Prints, only if the first argument is True-like."""
    if should_print:
        print(*args, **kwargs)


class StateError(RuntimeError):
    """Indicates that an object is not in the right state to perform an opteration."""
    def __init__(self, msg: str = ""):
        super().__init__(msg)
