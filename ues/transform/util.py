
import itertools
from typing import Any, List, Dict


def head(lst: List[Any]) -> Any:
    if not len(lst):
        raise ValueError("List is empty")
    return lst[0]


def dict_key(dictionary: Dict[Any, Any]) -> Any:
    if not isinstance(dictionary, dict):
        raise TypeError("Not a dict: " + str(dictionary))
    if not dictionary:
        raise ValueError("No entries")
    keys = list(dictionary.keys())
    if len(keys) > 1:
        raise ValueError("Ambigous call - dict contains multiple entries: " + str(dictionary))
    return next(iter(keys))


def dict_value(dictionary: Dict[Any, Any]) -> Any:
    if not isinstance(dictionary, dict):
        raise TypeError("Not a dict: " + str(dictionary))
    if not dictionary:
        raise ValueError("No entries")
    vals = list(dictionary.values())
    if len(vals) > 1:
        raise ValueError("Ambigous call - dict contains multiple entries: " + str(dictionary))
    return next(iter(vals))


def flatten(deep_lst: List[List[Any]]) -> List[Any]:
    return list(itertools.chain(*deep_lst))
