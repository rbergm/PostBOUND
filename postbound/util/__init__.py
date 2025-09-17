"""Contains utilities that are not specific to PostBOUND's domain of databases and query optimization."""

from . import collections, dicts, proc, stats, system, typing
from . import networkx as nx
from ._errors import InvariantViolationError, LogicError, StateError
from .collections import enlist, flatten, powerset, set_union, simplify
from .dicts import argmin, frozendict, hash_dict
from .jsonize import jsondict, to_json
from .logging import make_logger, timestamp
from .misc import DependencyGraph, Version, camel_case2snake_case
from .proc import run_cmd
from .stats import jaccard
from .system import open_files

__all__ = [
    "flatten",
    "enlist",
    "simplify",
    "set_union",
    "powerset",
    "collections",
    "hash_dict",
    "argmin",
    "frozendict",
    "dicts",
    "StateError",
    "LogicError",
    "InvariantViolationError",
    "jsondict",
    "to_json",
    "timestamp",
    "make_logger",
    "camel_case2snake_case",
    "Version",
    "DependencyGraph",
    "nx",
    "run_cmd",
    "proc",
    "jaccard",
    "stats",
    "open_files",
    "system",
    "typing",
]
