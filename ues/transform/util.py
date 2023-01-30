
import argparse
import collections
import collections.abc
import functools
import itertools
import json
import math
import numbers
import os
import pprint
import random
import sys
import threading
import typing
import warnings
from datetime import datetime
from typing import List, Dict, Set, Any, Iterable, Tuple, Union, Callable, IO

import networkx as nx
import numpy as np
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


def dict_invert(mapping: Dict[_K, List[_V]]) -> Dict[_V, List[_K]]:
    level1 = {tuple(vs): k for k, vs in mapping.items()}
    level2: Dict[_V, List[_K]] = {}
    for vs, k in level1.items():
        for v in vs:
            if v not in level2:
                level2[v] = [k]
            else:
                level2[v].append(k)
    return level2


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


def timestamp() -> str:
    return datetime.now().strftime("%y-%m-%d %H:%M:%S")


def make_logger(enabled: bool = True, *, file: IO[str] = sys.stderr, pretty: bool = False):
    def _log(*args, **kwargs):
        print(*args, file=file, **kwargs)

    def _dummy_log(*args, **kwargs):
        pass

    if pretty and enabled:
        return functools.partial(pprint.pprint, stream=file)

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


class AtomicInt(numbers.Integral):
    def __init__(self, value: int = 0):
        self._value = value
        self._lock = threading.Lock()

    def increment(self, by: int = 1) -> None:
        with self._lock:
            self._value += by

    def reset(self) -> None:
        with self._lock:
            self._value = 0

    def _get_value(self) -> int:
        with self._lock:
            return self._value

    def _set_value(self, value: int) -> None:
        with self._lock:
            self._value = value

    def _assert_integral(self, other: Any):
        if not isinstance(other, numbers.Integral):
            raise TypeError(f"Cannot add argument of type {type(other)} to object of type AtomicInt")

    def _unwrap_atomic(self, other: Any):
        return other._value if isinstance(other, AtomicInt) else other

    value = property(_get_value, _set_value)

    def __abs__(self) -> int:
        with self._lock:
            return abs(self._value)

    def __add__(self, other: Any) -> "AtomicInt":
        self._assert_integral(other)
        other = self._unwrap_atomic(other)
        with self._lock:
            return AtomicInt(self._value + other)

    def __and__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value & other

    def __ceil__(self) -> int:
        with self._lock:
            return math.ceil(self._value)

    def __eq__(self, other: object) -> bool:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value == other

    def __floor__(self) -> int:
        with self._lock:
            return math.floor(self._value)

    def __floordiv__(self, other: Any) -> int:
        other = self._unwrap_atomic(other)
        with self._lock:
            return AtomicInt(self._value // other)

    def __int__(self) -> int:
        with self._lock:
            return int(self._value)

    def __invert__(self) -> Any:
        with self._lock:
            return ~self._value

    def __le__(self, other: Any) -> bool:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value <= other

    def __lshift__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value << other

    def __lt__(self, other: Any) -> bool:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value < other

    def __mod__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value % other

    def __mul__(self, other: Any) -> "AtomicInt":
        self._assert_integral(other)
        other = self._unwrap_atomic(other)
        with self._lock:
            return AtomicInt(self._value * other)

    def __neg__(self) -> "AtomicInt":
        with self._lock:
            return AtomicInt(-self._value)

    def __or__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value | other

    def __pos__(self) -> Any:
        with self._lock:
            return +self.value

    def __pow__(self, exponent: Any, modulus: Union[Any, None] = ...) -> "AtomicInt":
        with self._lock:
            res = self._value ** exponent
            if res != int(res):
                raise ValueError(f"Power not support for type AtomicInt with argument {exponent}")
            return AtomicInt(res)

    def __radd__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other + self._value

    def __rand__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other + self._value

    def __rfloordiv__(self, other: Any) -> "AtomicInt":
        other = self._unwrap_atomic(other)
        with self._lock:
            return other // self._value

    def __rlshift__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other << self._value

    def __rmod__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other % self._value

    def __rmul__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other * self._value

    def __ror__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other | self._value

    def __round__(self, ndigits: Union[int, None] = None) -> int:
        with self._lock:
            return self._value

    def __rpow__(self, base: Any) -> Any:
        base = self._unwrap_atomic(base)
        with self._lock:
            return base ** self._value

    def __rrshift__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other >> self._value

    def __rshift__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value >> other

    def __rtruediv__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other / self._value

    def __rxor__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return other ^ self._value

    def __truediv__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value / other

    def __trunc__(self) -> int:
        with self._lock:
            return math.trunc(self._value)

    def __xor__(self, other: Any) -> Any:
        other = self._unwrap_atomic(other)
        with self._lock:
            return self._value ^ other

    def __hash__(self) -> int:
        with self._lock:
            return hash(self._value)

    def __repr__(self) -> str:
        with self._lock:
            return f"AtomicInt({self._value})"

    def __str__(self) -> str:
        with self._lock:
            return str(self._value)


class BoundedInt(numbers.Integral):
    @staticmethod
    def non_neg(value: int, *, allowed_max: Union[int, None] = None) -> "BoundedInt":
        return BoundedInt(value, allowed_min=0, allowed_max=allowed_max)

    def __init__(self, value: int, *, allowed_min: Union[int, None] = None, allowed_max: Union[int, None] = None):
        if not isinstance(value, int):
            raise TypeError(f"Only integer values allowed, but {type(value)} given!")
        if allowed_min is not None and allowed_max is not None and allowed_min > allowed_max:
            raise ValueError("Allowed minimum may not be larger than allowed maximum!")

        self._value = value
        self._allowed_min = allowed_min
        self._allowed_max = allowed_max

        # don't forget the first update!
        self._snap_to_min_max()

    def _snap_to_min_max(self) -> None:
        if self._allowed_min is not None and self._value < self._allowed_min:
            self._value = self._allowed_min
        if self._allowed_max is not None and self._value > self._allowed_max:
            self._value = self._allowed_max

    def _unwrap_atomic(self, value: Any) -> int:
        return value._value if isinstance(value, BoundedInt) else value

    def _get_value(self) -> int:
        return self._value

    def _set_value(self, value: int) -> None:
        if not isinstance(value, int):
            raise TypeError(f"Only integer values allowed, but {type(value)} given!")
        self._value = value
        self._snap_to_min_max()

    value = property(_get_value, _set_value)

    def __abs__(self) -> int:
        return abs(self._value)

    def __add__(self, other: Union[int, "BoundedInt"]) -> "BoundedInt":
        other_value = self._unwrap_atomic(other)
        return BoundedInt(self._value + other_value, allowed_min=self._allowed_min, allowed_max=self._allowed_max)

    def __and__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value & other_value

    def __ceil__(self) -> int:
        return self._value

    def __eq__(self, other: object) -> bool:
        other_value = self._unwrap_atomic(other)
        return self._value == other_value

    def __floor__(self) -> int:
        return self._value

    def __floordiv__(self, other: Any) -> int:
        other_value = self._unwrap_atomic(other)
        return self._value // other_value

    def __int__(self) -> int:
        return self._value

    def __invert__(self) -> Any:
        return ~self._value

    def __le__(self, other: Any) -> bool:
        other_value = self._unwrap_atomic(other)
        return self._value <= other_value

    def __lshift__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self.value << other_value

    def __lt__(self, other: Any) -> bool:
        other_value = self._unwrap_atomic(other)
        return self._value < other_value

    def __mod__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value % other_value

    def __mul__(self, other: Any) -> "BoundedInt":
        other_value = self._unwrap_atomic(other)
        return BoundedInt(self._value * other_value, allowed_min=self._allowed_min, allowed_max=self._allowed_max)

    def __neg__(self) -> "BoundedInt":
        return BoundedInt(-self._value, allowed_min=self._allowed_min, allowed_max=self._allowed_max)

    def __or__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value | other_value

    def __pos__(self) -> Any:
        return +self._value

    def __pow__(self, exponent: Any, modulus: Union[Any, None] = ...) -> "BoundedInt":
        res = self._value ** exponent
        if res != int(res):
            raise ValueError(f"Power not support for type BoundedInt with argument {exponent}")
        return BoundedInt(res, allowed_min=self._allowed_min, allowed_max=self._allowed_max)

    def __radd__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value + self._value

    def __rand__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value & self._value

    def __rfloordiv__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value // self._value

    def __rlshift__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value << self._value

    def __rmod__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value % self._value

    def __rmul__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value * self._value

    def __ror__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value | self._value

    def __round__(self, ndigits: Union[int, None] = None) -> int:
        return self._value

    def __rpow__(self, base: Any) -> Any:
        other_value = self._unwrap_atomic(base)
        return other_value ** self._value

    def __rrshift__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value >> self._value

    def __rshift__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value >> other_value

    def __rtruediv__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value / self._value

    def __rxor__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return other_value ^ self._value

    def __truediv__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value / other_value

    def __trunc__(self) -> int:
        return math.trunc(self._value)

    def __xor__(self, other: Any) -> Any:
        other_value = self._unwrap_atomic(other)
        return self._value ^ other_value

    def __hash__(self) -> int:
        return hash(self._value)

    def __repr__(self) -> str:
        return f"BoundedInt({self._value}; min={self._allowed_min}, max={self._allowed_max})"

    def __str__(self) -> str:
        return str(self._value)


class JsonizeEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if "__json__" in dir(obj):
            return obj.__json__()
        return json.JSONEncoder.default(self, obj)


def to_json(obj: Any, *args, **kwargs) -> str:
    if obj is None:
        return None
    kwargs.pop("cls", None)
    return json.dumps(obj, *args, cls=JsonizeEncoder, **kwargs)


def read_json(obj: Any) -> Any:
    if not obj or obj is np.nan:
        return {}
    return json.loads(obj)


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


def nx_random_walk(graph: nx.Graph):
    """A modified random walk implementation for networkx graphs.

    The modifications concern two specific areas: after each stop, the walk may jump to a node that is connected to one of the
    visited nodes. This node does not necessarily have to be connected to the current node. Secondly, if the graph contains
    multiple connected components, the walk will first explore one component before jumping to the next one.
    """
    shell_nodes = set()
    visited_nodes = set()

    total_n_nodes = len(graph.nodes)

    current_node = random.choice(list(graph.nodes))
    visited_nodes.add(current_node)
    yield current_node

    while len(visited_nodes) < total_n_nodes:
        shell_nodes |= set(n for n in graph.adj[current_node].keys() if n not in visited_nodes)
        if not shell_nodes:
            # we have multiple connected components and need to jump into the other component
            current_node = random.choice([n for n in graph.nodes if n not in visited_nodes])
            visited_nodes.add(current_node)
            yield current_node
            continue

        current_node = random.choice(list(shell_nodes))
        shell_nodes.remove(current_node)
        visited_nodes.add(current_node)
        yield current_node


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
