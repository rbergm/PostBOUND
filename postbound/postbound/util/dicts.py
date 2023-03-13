"""Contains utilities to access and modify dictionaries more conveniently."""
from __future__ import annotations

import collections
import collections.abc
import itertools
import numbers
import typing
import warnings
from typing import Callable, Optional

T = typing.TypeVar("T")
K = typing.TypeVar("K")
V = typing.TypeVar("V")


def key(dictionary: dict[K, V]) -> K:
    """Provides the key of a dictionary with just 1 item.

    `key({'a': 1}) = 'a'`
    """
    if not len(dictionary) == 1:
        raise ValueError("Dictionary must contain exactly 1 entry, not " + str(len(dictionary)))
    for k in dictionary:
        return k


def value(dictionary: dict[K, V]) -> V:
    """Provides the value of a dictionary with just 1 item.

    `value({'a': 1}) = 1`
    """
    if not len(dictionary) == 1:
        raise ValueError("Dictionary must contain exactly 1 entry, not " + str(len(dictionary)))
    for v in dictionary.values():
        return v


def merge(a: dict[K, V], b: dict[K, V], *, updater: Optional[Callable[[K, V, V], V]] = None) -> dict[K, V]:
    """Creates a new dict containing all key/values pairs from both argument dictionaries.

    If keys overlap, entries from dictionary `b` will take priority, unless an `update` method is given.
    If `update` is given, and `a[k] = v` and `b[k] = v'` (i.e. both `a` and `b` share a key `k`) the merged dictionary
    will contain the result of `update(k, v, v')` as entry for `k`.

    Note that as of Python 3.9, such a method was added to dictionaries as well (via the `|=` syntax). Our current
    implementation is not optimized for larger dictionaries and will probably have a pretty bad performance on such
    input data.
    """
    if not updater:
        return dict([*a.items()] + [*b.items()])
    else:
        merged = dict(a)
        for k, v in b.items():
            if k in merged:
                merged[k] = updater(k, merged[k], v)
            else:
                merged[k] = v
        return merged


def update(dictionary: dict[K, V], updater: Callable[[K, V], T]) -> dict[K, T]:
    """Creates a new dict by calling update on each key/value pair on the old dict, retaining its keys."""
    return {k: updater(k, v) for k, v in dictionary.items()}


def explode(dictionary: dict[K, list[V]]) -> list[tuple[K, V]]:
    """Transforms dicts mapping keys to lists of values to a list of key/value pairs."""
    values = []
    for k, dict_values in dictionary.items():
        values.extend(zip(itertools.cycle([k]), dict_values))
    return values


def hash_dict(dictionary: dict[K, V]) -> int:
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


def generate_multi(entries: list[tuple[K, V]]) -> dict[K, list[V]]:
    """Generates a multi-dict based on its entries.

    Each key can occur multiple times and values will be aggregated in a list.
    """
    collector = collections.defaultdict(list)
    for k, v in entries:
        collector[k].append(v)
    return dict(collector)


def reduce_multi(multi_dict: dict[K, list[V]], reduction: Callable[[K, list[V]], V]) -> dict[K, V]:
    """Ungroups a multi-dict by aggregating the values based on key and values."""
    return {k: reduction(k, vs) for k, vs in multi_dict.items()}


def invert_multi(mapping: dict[K, list[V]]) -> dict[V, list[K]]:
    """Inverts the `key -> values` mapping of a dict to become `value -> keys` instead.

    Supppose a multi-dict has the following contents: `{'a': [1, 2], 'b': [2, 3]}`.
    Calling `invert` transforms this mapping to `{1: ['a'], 2: ['a', 'b'], 3: ['b']}`.
    """
    level1 = {tuple(vs): k for k, vs in mapping.items()}
    level2: dict[V, list[K]] = {}
    for vs, k in level1.items():
        for v in vs:
            if v not in level2:
                level2[v] = [k]
            else:
                level2[v].append(k)
    return level2


def invert(mapping: dict[K, V]) -> dict[V, K]:
    """Inverts the `key -> value` mapping of a dict to become `value -> key` instead.

    In contrast to `invert_multi` this does not handle duplicate values (which leads to duplicate keys), nor does it
    process the original values of `mapping` in any way.

    Basically, this function is just a better-readable shortcut for `{v: k for k, v in d.items()}`.
    """
    return {v: k for k, v in mapping.items()}


def argmin(mapping: dict[K, numbers.Number]) -> K:
    """
    For a dict mapping keys to numeric types, returns the key `k` with minimum value `v`, s.t. for all keys `k'` with
    values `v'` it holds that `v <= v'`.
    """
    return min(mapping, key=mapping.get)
