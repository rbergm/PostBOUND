
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


def flatten(deep_lst: List[List[Any]], *, recursive=False) -> List[Any]:
    deep_lst = [[deep_elem] if not isinstance(deep_elem, list) else deep_elem for deep_elem in deep_lst]
    flattened = list(itertools.chain(*deep_lst))
    if recursive and any(isinstance(deep_elem, list) for deep_elem in flattened):
        return flatten(flattened, recursive=True)
    return flattened


def enlist(obj: Any) -> List[Any]:
    return obj if isinstance(obj, list) else [obj]


def represents_number(val: str) -> bool:
    try:
        float(val)
    except (TypeError, ValueError):
        return False
    return True
