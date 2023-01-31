"""Contains utilties to access and modify dictionaries more conveniently."""

import collections
import itertools
import numbers
import typing
import warnings
from typing import Dict, List, Callable, Tuple

_T = typing.TypeVar("_T")
_K = typing.TypeVar("_K")
_V = typing.TypeVar("_V")


def key(dictionary: Dict[_K, _V]) -> _K:
    """Provides the key of a dictionary with just 1 item.

    `key({'a': 1}) = 'a'`
    """
    if not len(dictionary) == 1:
        raise ValueError("Dictionary must contain exactly 1 entry, not " + str(len(dictionary)))
    for k in dictionary:
        return k

def value(dictionary: Dict[_K, _V]) -> _V:
    """Provides the value of a dictionary with just 1 item.

    `value({'a': 1}) = 1`
    """
    if not len(dictionary) == 1:
        raise ValueError("Dictionary must contain exactly 1 entry, not " + str(len(dictionary)))
    for v in dictionary.values():
        return v


def merge(a: Dict[_K, _V], b: Dict[_K, _V], *, update: Callable[[_K, _V, _V], _V] = None) -> Dict[_K, _V]:
    """Creates a new dict containing all key/values pairs from both argument dictionaries.

    If keys overlap, entries from dictionary `b` will take priority, unless an `update` method is given.
    If `update` is given, and `a[k] = v` and `b[k] = v'` (i.e. both `a` and `b` share a key `k`) the merged dictionary
    will contain the result of `update(k, v, v')` as entry for `k`.

    Note that as of Python 3.9, such a method was added to dictionaries as well (via the `|=` syntax). Our current
    implementation is not optimized for larger dictionaries and will probably have a pretty bad performance on such
    input data.
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


def update(dictionary: Dict[_K, _V], update: Callable[[_K, _V], _T]) -> Dict[_K, _T]:
    """Creates a new dict by calling update on each key/value pair on the old dict, retaining its keys."""
    return {key: update(val) for key, val in dictionary.items()}


def explode(dictionary: Dict[_K, List[_V]]) -> List[Tuple[_K, _V]]:
    """Transforms dicts mapping keys to lists of values to a list of key/value pairs."""
    values = []
    for key, dict_values in dictionary.items():
        values.extend(zip(itertools.cycle([key]), dict_values))
    return values


def hash_dict(dictionary: Dict[_K, _V]) -> int:
    """Calculates a hash value based on the current dict contents (keys and values)."""
    keys = list(dictionary.keys())
    values = []
    for val in dictionary.values():
        if isinstance(val, collections.abc.Hashable):
            values.append(hash(val))
        elif isinstance(val, list) or isinstance(val, set):
            values.append(hash(tuple(val)))
        elif isinstance(val, dict):
            values.append(hash_dict(val))
        else:
            warnings.warn("Unhashable type, skipping: " + type(val))
    keys_hash = hash(tuple(keys))
    values_hash = hash(tuple(values))
    return hash((keys_hash, values_hash))


def generate_multi(entries: List[Tuple[_K, _V]]) -> Dict[_K, List[_V]]:
    """Generates a multi-dict based on its entries.

    Each key can occur multiple times and values will be aggregated in a list.
    """
    collector = collections.defaultdict(list)
    for key, value in entries:
        collector[key].append(value)
    return dict(collector)


def reduce_multi(multi_dict: Dict[_K, List[_V]], reduction: Callable[[_K, List[_V]], _V]) -> Dict[_K, _V]:
    """Ungroups a multi-dict by aggregating the values based on key and values."""
    return {key: reduction(key, values) for key, values in multi_dict.items()}


def invert(mapping: Dict[_K, List[_V]]) -> Dict[_V, List[_K]]:
    """Inverts the `key -> values` mapping of a dict to become `value -> keys` instead.

    Supppose a multi-dict has the following contents: `{'a': [1, 2], 'b': [2, 3]}`.
    Calling `invert` transforms this mapping to `{1: ['a'], 2: ['a', 'b'], 3: ['b']}`.
    """
    level1 = {tuple(vs): k for k, vs in mapping.items()}
    level2: Dict[_V, List[_K]] = {}
    for vs, k in level1.items():
        for v in vs:
            if v not in level2:
                level2[v] = [k]
            else:
                level2[v].append(k)
    return level2


def argmin(mapping: Dict[_K, numbers.Number]) -> _K:
    """
    For a dict mapping keys to numeric types, returns the key `k` with minimum value `v`, s.t. for all keys `k'` with
    values `v'` it holds that `v <= v'`.
    """
    return min(mapping, key=mapping.get)
