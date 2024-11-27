"""Fundamental types for the query abstraction layer. This includes references to tables as well as columns.

All reference types  are designed as immutable data objects whose content cannot be changed. Any forced modifications
will break the entire query abstraction layer and lead to unpredictable behaviour.
"""
from __future__ import annotations

from typing import Optional

from postbound.util import errors, jsonize


class TableReference(jsonize.Jsonizable):
    """A table reference represents a database table.

    It can either be a physical table, a CTE, or an entirely virtual query created via subqueries. Note that a table
    reference is indeed just a reference and not a 1:1 "representation" since each table can be sourced multiple times
    in a query. Therefore, in addition to the table name, each instance can optionally also contain an alias to
    distinguish between different references to the same table. In case of virtual tables, the full name will usually be empty
    and only the alias set. An exception are table references that refer to CTEs: their full name is set to the CTE name, the
    alias to the alias from the FROM clause (if present) and the table is still treated as virtual.

    Table references can be sorted lexicographically. All instances should be treated as immutable objects.

    Parameters
    ----------
    full_name : str
        The name of the table, corresponding to the name of a physical database table (or a view)
    alias : str, optional
        Alternative name that is in queries to refer to the table, or to refer to the same table multiple times.
        Defaults to an empty string
    virtual : bool, optional
        Whether the table is virtual or not. As a rule of thumb, virtual tables cannot be part of a ``FROM`` clause on
        their own, but need some sort of context. For example, the alias of a subquery is typically represented as a
        virtual table in PostBOUND. One cannot directly reference that alias in a ``FROM`` clause, without also
        specifying the subquery. Defaults to ``False`` since most tables will have direct physical counterparts.

    Raises
    ------
    ValueError
        If neither full name nor an alias are provided
    """

    @staticmethod
    def create_virtual(alias: str, *, full_name: str = "") -> TableReference:
        """Generates a new virtual table reference with the given alias.

        Parameters
        ----------
        alias : str
            The alias of the virtual table. Cannot be ``None``.
        full_name : str, optional
            An optional full name for the entire table. This is mostly used to create references to CTE tables.

        Returns
        -------
        TableReference
            The virtual table reference
        """
        return TableReference(full_name, alias, True)

    def __init__(self, full_name: str, alias: str = "", virtual: bool = False) -> None:
        if not full_name and not alias:
            raise ValueError("Full name or alias required")
        self._full_name = full_name if full_name else ""
        self._alias = alias if alias else ""
        self._virtual = virtual
        self._hash_val = hash((full_name, alias))

    @property
    def full_name(self) -> str:
        """Get the full name of this table. If empty, alias is guaranteed to be set.

        Returns
        -------
        str
            The name of the table
        """
        return self._full_name

    @property
    def alias(self) -> str:
        """Get the alias of this table. If empty, the full name is guaranteed to be set.

        The precise semantics of alias usage differ from database system to system. For example, in Postgres an alias
        shadows the original table name, i.e. once an alias is specified, it *must* be used to reference to the table
        and its columns.

        Returns
        -------
        str
            The alias of the table
        """
        return self._alias

    @property
    def virtual(self) -> bool:
        """Checks whether this table is virtual. In this case, only the alias and not the full name is set.

        Returns
        -------
        bool
            Whether this reference describes a virtual table
        """
        return self._virtual

    def identifier(self) -> str:
        """Provides a shorthand key that columns can use to refer to this table reference.

        For example, a table reference for ``movie_companies AS mc`` would have ``mc`` as its identifier (i.e. the
        alias), whereas a table reference without an alias such as ``company_type`` would provide the full table name
        as its identifier, i.e. ``company_type``.

        Returns
        -------
        str
            The shorthand
        """
        return self.alias if self.alias else self.full_name

    def drop_alias(self) -> TableReference:
        """Removes the alias from the current table if there is one. Returns the tabel as-is otherwise.

        Returns
        -------
        TableReference
            This table, but without an alias. Since table references are immutable, the original reference is not
            modified

        Raises
        ------
        errors.StateError
            If this table is a virtual table, since virtual tables only have an alias and no full name.
        """
        if self.virtual:
            raise errors.StateError("An alias cannot be dropped from a virtual table!")
        return TableReference(self.full_name)

    def with_alias(self, alias: str) -> TableReference:
        """Creates a new table reference for the same table but with a different alias.

        Parameters
        ----------
        alias : str
            The new alias

        Returns
        -------
        TableReference
            The updated table reference

        Raises
        ------
        errors.StateError
            If the current table does not have a full name.
        """
        if not self.full_name:
            raise errors.StateError("Cannot add an alias to a table without full name")
        return TableReference(self.full_name, alias, self.virtual)

    def __json__(self) -> object:
        return {"full_name": self._full_name, "alias": self._alias}

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TableReference):
            return NotImplemented
        return self.identifier() < other.identifier()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, __value: object) -> bool:
        return (isinstance(__value, type(self))
                and self._full_name == __value._full_name
                and self._alias == __value._alias)

    def __repr__(self) -> str:
        return f"TableReference(full_name='{self.full_name}', alias='{self.alias}', virtual={self.virtual})"

    def __str__(self) -> str:
        if self.full_name and self.alias:
            return f"{self.full_name} AS {self.alias}"
        elif self.alias:
            return self.alias
        elif self.full_name:
            return self.full_name
        else:
            return "[UNKNOWN TABLE]"


class ColumnReference(jsonize.Jsonizable):
    """A column reference represents a specific column of a specific database table.

    This reference always consists of the name of the "physical" column (see below for special cases). In addition,
    each column can be bound to the table to which it belongs by providing the associated table reference.

    Column references can be sorted lexicographically and are designed as immutable data objects.

    Parameters
    ----------
    name : str
        The name of the column. Cannot be empty.
    table : Optional[TableReference], optional
        The table which provides the column. Can be ``None`` if the table is unknown.

    Raises
    ------
    ValueError
        If the name is empty (or ``None``)

    Notes
    -----
    A number of special cases arise when dealing with subqueries and common table expressions. The first one is the
    fact that columns can be bound to virtual tables, e.g. if they are exported by subqueries, etc. In the same vein,
    columns also do not always need to refer directly to physical columns. Consider the following example query:

    ::

        WITH cte_table AS (SELECT foo.f_id, foo.a + foo.b AS 'sum' FROM foo)
        SELECT *
        FROM bar JOIN cte_table ON bar.b_id = cte_table.f_id
        WHERE cte_table.sum < 42

    In this case, the CTE exports a column ``sum`` that is constructed based on two "actual" columns. Hence, the sum
    column itself does not have any physical representation but will be modelled as a column reference nevertheless.
    """

    def __init__(self, name: str, table: Optional[TableReference] = None) -> None:
        if not name:
            raise ValueError("Column name is required")
        self._name = name
        self._table = table
        self._hash_val = hash((self._name, self._table))

    @property
    def name(self) -> str:
        """Get the name of this column. This is guaranteed to be set and will never be empty

        Returns
        -------
        str
            The name
        """
        return self._name

    @property
    def table(self) -> Optional[TableReference]:
        """Get the table to which this column belongs, if specified.

        Returns
        -------
        Optional[TableReference]
            The table or ``None``. The table can be an arbitrary reference, i.e. virtual or physical.
        """
        return self._table

    def is_bound(self) -> bool:
        """Checks, whether this column is bound to a table.

        Returns
        -------
        bool
            Whether a valid table reference is set
        """
        return self.table is not None

    def belongs_to(self, table: TableReference) -> bool:
        """Checks, whether the column is part of the given table.

        This check does not consult the schema of the actual database or the like, it merely matches the given table
        reference with the `table` attribute of this column.

        Parameters
        ----------
        table : TableReference
            The table to check

        Returns
        -------
        bool
            Whether the table's column is the same as the given one
        """
        return table == self.table

    def bind_to(self, table: TableReference) -> ColumnReference:
        """Binds this column to a new table.

        Parameters
        ----------
        table : TableReference
            The new table

        Returns
        -------
        ColumnReference
            The updated column reference, the original reference is not modified.
        """
        return ColumnReference(self.name, table)

    def __json__(self) -> object:
        return {"name": self._name, "table": self._table}

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ColumnReference):
            return NotImplemented
        if self.table == other.table:
            return self.name < other.name
        if not self.table:
            return True
        if not other.table:
            return False
        return self.table < other.table

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.name == other.name and self.table == other.table

    def __repr__(self) -> str:
        return f"ColumnReference(name='{self.name}', table={repr(self.table)})"

    def __str__(self) -> str:
        if self.table and self.table.alias:
            return f"{self.table.alias}.{self.name}"
        elif self.table and self.table.full_name:
            return f"{self.table.full_name}.{self.name}"
        return self.name


class UnboundColumnError(errors.StateError):
    """Indicates that a column is required to be bound to a table, but the provided column was not.

    Parameters
    ----------
    column : ColumnReference
        The column without the necessary table binding
    """

    def __init__(self, column: ColumnReference) -> None:
        super().__init__("Column is not bound to any table: " + str(column))
        self.column = column


class VirtualTableError(errors.StateError):
    """Indicates that a table is required to correspond to a physical table, but the provided reference was not.

    Parameters
    ----------
    table : TableReference
        The virtual table
    """

    def __init__(self, table: TableReference) -> None:
        super().__init__("Table is virtual: " + str(table))
        self.table = table
