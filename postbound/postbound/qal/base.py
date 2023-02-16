"""Fundamental types for the query abstraction layer."""

from __future__ import annotations

from dataclasses import dataclass

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
    virtual: bool = False

    def identifier(self) -> str:
        return self.alias if self.alias else self.full_name

    def __lt__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.full_name == other.full_name:
            return self.alias < other.alias
        return self.full_name < other.full_name

    def __hash__(self) -> int:
        return hash((self.full_name, self.alias, self.virtual))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self)) and self.full_name == other.full_name
                and self.alias == other.alias and self.virtual == other.virtual)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.full_name and self.alias:
            return f"{self.full_name} AS {self.alias}"
        elif self.alias:
            return self.alias
        elif self.full_name:
            return self.full_name
        else:
            return "[UNKNOWN TABLE]"


@dataclass
class ColumnReference:
    name: str
    table: TableReference | None = None
    redirect: ColumnReference | None = None

    def resolve(self):
        return self.redirect.resolve() if self.redirect else self

    def __lt__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        if self.table == other.table:
            return self.name < other.name
        return self.table < other.table

    def __hash__(self) -> int:
        return hash(tuple([self.name, self.table]))

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.name == other.name and self.table == other.table

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.table and self.table.alias:
            return f"{self.table.alias}.{self.name}"
        elif self.table and self.table.full_name:
            return f"{self.table.full_name}.{self.name}"
        return self.name


class UnboundColumnError(errors.StateError):
    def __init__(self, column: ColumnReference) -> None:
        super().__init__("Column is not bound to any table: " + str(column))
        self.column = column