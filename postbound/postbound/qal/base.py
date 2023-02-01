"""Fundamental types for the query abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from postbound.util import errors


@dataclass
class TableReference:
    """A table reference represents a database table.

    It can either be a physical table, a CTE, or an entirely virtual query created via subqueries. Note that a table
    reference is indeed just a reference and not a 1:1 "representation" since each table can be sourced multiple times
    in a query. Therefore, in addition to the table name, each instance can optionally also contain an alias to
    distinguish between different references to the same table.
    """
    full_name: str
    alias: str = ""

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.full_name == other.full_name:
            return self.alias < other.alias
        return self.full_name < other.full_name


@dataclass
class ColumnReference:
    name: str
    table: Union[TableReference, None] = None

    def __lt__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.table == other.table:
            return self.name < other.name
        return self.table < other.table


class UnboundColumnError(errors.StateError):
    def __init__(self, column: ColumnReference) -> None:
        super().__init__("Column is not bound to any table: " + str(column))
        self.column = column
