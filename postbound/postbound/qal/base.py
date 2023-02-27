"""Fundamental types for the query abstraction layer. This includes references to tables as well as columns."""

from __future__ import annotations

from dataclasses import dataclass

from postbound.util import errors


@dataclass
class TableReference:
    """A table reference represents a database table.

    It can either be a physical table, a CTE, or an entirely virtual query created via subqueries. Note that a table
    reference is indeed just a reference and not a 1:1 "representation" since each table can be sourced multiple times
    in a query. Therefore, in addition to the table name, each instance can optionally also contain an alias to
    distinguish between different references to the same table. In case of virtual tables, the full name will be empty
    and only the alias set.

    Table references can be sorted lexicographically.
    """
    full_name: str
    alias: str = ""
    virtual: bool = False

    @staticmethod
    def create_virtual(alias: str) -> TableReference:
        """Generates a new virtual table reference with the given alias."""
        return TableReference("", alias, True)

    def identifier(self) -> str:
        """Provides a shorthand key that columns can use to refer to this table reference.

        For example, a table reference for `movie_companies AS mc` would have `mc` as its identifier (i.e. the alias),
        whereas a table reference without an alias such as `company_type` would provide the full table name as its
        identifier, i.e. `company_type`.
        """
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
    """A column reference represents a specific column of a specific database table.

    This reference always consists of the name of the physical table. In addition, each column can be bound to the
    table to which it belongs by providing the associated table reference.

    Since subqueries can export specific columns, references do not need to be physical tables. Instead, they can
    refer to columns of virtual tables which export their columns under different names than the original (physical)
    column. To accommodate for such situations, columns references can redirect to other column references. Use the
    `resolve` method to retrieve the actual column reference (which will most likely correspond to a physical column
    of a physical table).

    Column references can be sorted lexicographically.
    """
    name: str
    table: TableReference | None = None
    redirect: ColumnReference | None = None

    def resolve(self) -> ColumnReference:
        """Traverse the column redirections until a non-redirecting reference is found."""
        return self.redirect.resolve() if self.redirect else self

    def is_bound(self) -> bool:
        """Checks, whether this column is bound to a table."""
        return self.table is not None

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
    """Indicates that a column is required to be bound to a table, but the provided column was not."""

    def __init__(self, column: ColumnReference) -> None:
        super().__init__("Column is not bound to any table: " + str(column))
        self.column = column


class VirtualTableError(errors.StateError):
    """Indicates that a table is required to correspond to a physical table, but the provided reference was not."""

    def __init__(self, table: TableReference) -> None:
        super().__init__("Table is virtual: " + str(table))
        self.table = table
