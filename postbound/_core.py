
from __future__ import annotations

import re
from enum import Enum
from typing import Optional, TypeVar

from .util.errors import StateError


T = TypeVar("T")
"""Typed expressions use this generic type variable."""


VisitorResult = TypeVar("VisitorResult")
"""Result of visitor invocations."""


Cost = float
"""Type alias for a cost estimate."""

Cardinality = float
"""Type alias for a cardinality estimate.

Despite the type alias, cardinalities should not be ordinary floating point numbers but rather integers. We only alias float
instead of int to allow *NaN* for missing cardinalities as well as *inf* for infinite cardinalities.
Therefore, in order to check for valid cardinalities, ``math.isnan()`` should be used (or ``math.isinf()``, depneding on the
context).

We could be more precise and introduce our own cardinality type (which allows int | NaN | inf), but for now this is good enough
and allows to easily return integers.
"""


class ScanOperator(Enum):
    """The scan operators supported by PostBOUND.

    These can differ from the scan operators that are actually available in the selected target database system. The individual
    operators are chosen because they are supported by a wide variety of database systems and they are sufficiently different
    from each other.
    """
    SequentialScan = "Seq. Scan"
    IndexScan = "Idx. Scan"
    IndexOnlyScan = "Idx-only Scan"
    BitmapScan = "Bitmap Scan"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.value < other.value


class JoinOperator(Enum):
    """The join operators supported by PostBOUND.

    These can differ from the join operators that are actually available in the selected target database system. The individual
    operators are chosen because they are supported by a wide variety of database systems and they are sufficiently different
    from each other.
    """
    NestedLoopJoin = "NLJ"
    HashJoin = "Hash Join"
    SortMergeJoin = "Sort-Merge Join"
    IndexNestedLoopJoin = "Idx. NLJ"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.value < other.value


class IntermediateOperator(Enum):
    """The intermediate operators supported by PostBOUND.

    Intermediate operators are those that do not change the contents of their input relation, but only the way in which it is
    available. For example, a sort operator changes the order of the tuples.
    """

    Sort = "Sort"
    Memoize = "Memoize"
    Materialize = "Materialize"

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.value < other.value


PhysicalOperator = ScanOperator | JoinOperator | IntermediateOperator
"""Supertype to model all physical operators supported by PostBOUND.

These can differ from the operators that are actually available in the selected target database system.
"""

_IdentifierPattern = re.compile(r"^[a-z_][a-z0-9_\$]*$")
"""Regular expression to check for valid identifiers.

In line with Postgres' way of name resolution, we only permit identifiers with lower case characters. This forces all
identifiers which contain at least one upper case character to be quoted.

References
----------
- Postgres documentation on identifiers: https://www.postgresql.org/docs/current/sql-syntax-lexical.html#SQL-SYNTAX-IDENTIFIERS
- Regex tester: https://regex101.com/r/TtrNQg/1
"""

SqlKeywords = frozenset({
    "ALL", "AND", "ANY", "ARRAY", "AS", "ASC", "ASYMMETRIC", "BINARY", "BOTH", "CASE", "CAST", "CHECK", "COLLATE",
    "COLUMN", "CONSTRAINT", "CREATE", "CROSS", "CURRENT_CATALOG", "CURRENT_DATE", "CURRENT_ROLE", "CURRENT_SCHEMA",
    "CURRENT_TIME", "CURRENT_TIMESTAMP", "CURRENT_USER", "DEFAULT", "DEFERRABLE", "DESC", "DISTINCT", "DO", "ELSE",
    "END", "EXCEPT", "FALSE", "FETCH", "FOR", "FOREIGN", "FROM", "FULL", "GRANT", "GROUP", "HAVING", "ILIKE", "IN",
    "INITIALLY", "INNER", "INTERSECT", "INTO", "IS", "JOIN", "LATERAL", "LEADING", "LEFT", "LIKE", "LIMIT", "LOCALTIME",
    "LOCALTIMESTAMP", "NATURAL", "NOT", "NULL", "OFFSET", "ON", "ONLY", "OR", "ORDER", "OUTER", "OVERLAPS", "PLACING",
    "PRIMARY", "REFERENCES", "RETURNING", "RIGHT", "SELECT", "SESSION_USER", "SIMILAR", "SOME", "SYMMETRIC", "TABLE",
    "THEN", "TO", "TRAILING", "TRUE", "UNION", "UNIQUE", "USER", "USING", "VARIADIC", "VERBOSE", "WHEN", "WHERE",
    "WINDOW", "WITH"
})
"""An (probably incomplete) list of reserved SQL keywords that must be quoted before being used as identifiers."""


def quote(identifier: str) -> str:
    """Quotes an identifier if necessary.

    Valid identifiers can be used as-is, e.g. *title* or *movie_id*. Invalid identifiers will be wrapped in quotes, such as
    *"movie title"* or *"movie-id"*.

    Parameters
    ----------
    identifier : str
        The identifier to quote. Note that empty strings are treated as valid identifiers.

    Returns
    -------
    str
        The identifier, potentially wrapped in quotes.
    """
    if not identifier:
        return ""
    valid_identifier = _IdentifierPattern.fullmatch(identifier) and identifier.upper() not in SqlKeywords
    return identifier if valid_identifier else f'"{identifier}"'


def normalize(identifier: str) -> str:
    """Generates a normalized version of an identifier.

    Normalization is based on the Postgres rules of performing all comparisons in a case-insensitive manner.

    Parameters
    ----------
    identifier : str
        The identifier to normalize. Notice that empty strings can be normalized as well (without doing anything).

    Returns
    -------
    str
        The normalized identifier
    """
    return identifier.lower()


class TableReference:
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
        Whether the table is virtual or not. As a rule of thumb, virtual tables cannot be part of a *FROM* clause on
        their own, but need some sort of context. For example, the alias of a subquery is typically represented as a
        virtual table in PostBOUND. One cannot directly reference that alias in a *FROM* clause, without also
        specifying the subquery. Defaults to *False* since most tables will have direct physical counterparts.
    schema : str, optional
        The schema in which the table is located. Defaults to an empty string if the table is in the default schema or the
        schema is unknown.

    Raises
    ------
    ValueError
        If neither full name nor an alias are provided, or if a schema without a full name is provided.
    """

    @staticmethod
    def create_virtual(alias: str, *, full_name: str = "", schema: str = "") -> TableReference:
        """Generates a new virtual table reference with the given alias.

        Parameters
        ----------
        alias : str
            The alias of the virtual table. Cannot be *None*.
        full_name : str, optional
            An optional full name for the entire table. This is mostly used to create references to CTE tables.
        schema : str, optional
            The schema in which the table is located. Defaults to an empty string if the table is in the default schema or the
            schema is unknown.

        Returns
        -------
        TableReference
            The virtual table reference
        """
        return TableReference(full_name, alias, virtual=True, schema=schema)

    def __init__(self, full_name: str, alias: str = "", *, virtual: bool = False, schema: str = "") -> None:
        if not full_name and not alias:
            raise ValueError("Full name or alias required")
        if schema and not full_name:
            raise ValueError("Schema can only be set if a full name is provided")

        self._schema = schema if schema else ""
        self._full_name = full_name if full_name else ""
        self._alias = alias if alias else ""
        self._virtual = virtual

        self._identifier = self._alias if self._alias else self._full_name

        self._normalized_schema = normalize(self._schema)
        self._normalized_full_name = normalize(self._full_name)
        self._normalized_alias = normalize(self._alias)
        self._nomalized_identifier = normalize(self._identifier)
        self._hash_val = hash((self._normalized_full_name, self._normalized_alias, self._normalized_schema))

        table_txt = f"{quote(self._schema)}.{quote(self._full_name)}" if self._schema else quote(self._full_name)
        if table_txt and self._alias:
            self._sql_repr = f"{table_txt} AS {quote(self._alias)}"
        elif self._alias:
            self._sql_repr = quote(self._alias)
        elif self._full_name:
            self._sql_repr = table_txt
        else:
            raise ValueError("Full name or alias required")

    __match_args__ = ("full_name", "alias", "virtual", "schema")

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

    @property
    def schema(self) -> str:
        """Get the schema in which this table is located.

        Returns
        -------
        str
            The schema or an empty string if the schema is either unknown or the table is located in the default schema.
        """

        return self._schema

    def identifier(self) -> str:
        """Provides a shorthand key that columns can use to refer to this table reference.

        For example, a table reference for *movie_companies AS mc* would have *mc* as its identifier (i.e. the
        alias), whereas a table reference without an alias such as *company_type* would provide the full table name
        as its identifier, i.e. *company_type*.

        Returns
        -------
        str
            The shorthand
        """
        return self._identifier

    def qualified_name(self) -> str:
        """Provides the fully qualified name (i.e. including the schema) of this table.

        Notice that virtual tables do not have a qualified name, since they do not correspond to a physical table.

        Returns
        -------
        str
            The qualified name, quoted as necessary.
        """
        if self.virtual:
            raise VirtualTableError(f"Table {self} does not have a qualified name.")
        return f"{quote(self._schema)}.{quote(self._full_name)}" if self._schema else quote(self._full_name)

    def drop_alias(self) -> TableReference:
        """Removes the alias from the current table if there is one. Returns the tabel as-is otherwise.

        Returns
        -------
        TableReference
            This table, but without an alias. Since table references are immutable, the original reference is not
            modified

        Raises
        ------
        StateError
            If this table is a virtual table, since virtual tables only have an alias and no full name.
        """
        if self.virtual:
            raise StateError("An alias cannot be dropped from a virtual table!")
        return TableReference(self.full_name, schema=self._schema)

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
        StateError
            If the current table does not have a full name.
        """
        if not self.full_name:
            raise StateError("Cannot add an alias to a table without full name")
        return TableReference(self.full_name, alias, virtual=self.virtual)

    def make_virtual(self) -> TableReference:
        """Creates a new virtual table reference for the same table.

        Returns
        -------
        TableReference
            The updated table reference
        """
        return TableReference(self.full_name, self.alias, virtual=True, schema=self._schema)

    def update(self, *, full_name: Optional[str] = None, alias: Optional[str] = None, virtual: Optional[bool] = None,
               schema: Optional[str] = "") -> TableReference:
        full_name = self._full_name if full_name is None else full_name
        alias = self._alias if alias is None else alias
        virtual = self._virtual if virtual is None else virtual
        schema = self._schema if schema is None else schema
        return TableReference(full_name, alias, virtual=virtual, schema=schema)

    def __json__(self) -> object:
        return {"full_name": self._full_name, "alias": self._alias, "virtual": self._virtual, "schema": self._schema}

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, TableReference):
            return NotImplemented
        return self._nomalized_identifier < other._nomalized_identifier

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self._normalized_full_name == other._normalized_full_name
                and self._normalized_alias == other._normalized_alias
                and self._normalized_schema == other._normalized_schema)

    def __repr__(self) -> str:
        return (f"TableReference(full_name='{self.full_name}', alias='{self.alias}', "
                f"virtual={self.virtual}, schema='{self.schema}')")

    def __str__(self) -> str:
        return self._sql_repr


class ColumnReference:
    """A column reference represents a specific column of a specific database table.

    This reference always consists of the name of the "physical" column (see below for special cases). In addition,
    each column can be bound to the table to which it belongs by providing the associated table reference.

    Column references can be sorted lexicographically and are designed as immutable data objects.

    Parameters
    ----------
    name : str
        The name of the column. Cannot be empty.
    table : Optional[TableReference], optional
        The table which provides the column. Can be *None* if the table is unknown.

    Raises
    ------
    ValueError
        If the name is empty (or *None*)

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

    In this case, the CTE exports a column *sum* that is constructed based on two "actual" columns. Hence, the sum
    column itself does not have any physical representation but will be modelled as a column reference nevertheless.
    """

    def __init__(self, name: str, table: Optional[TableReference] = None) -> None:
        if not name:
            raise ValueError("Column name is required")
        self._name = name
        self._table = table
        self._normalized_name = normalize(self._name)
        self._hash_val = hash((self._normalized_name, self._table))

        if self._table:
            self._sql_repr = f"{quote(self._table.identifier())}.{quote(self._name)}"
        else:
            self._sql_repr = quote(self._name)

    __match_args__ = ("name", "table")

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
            The table or *None*. The table can be an arbitrary reference, i.e. virtual or physical.
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

    def as_unbound(self) -> ColumnReference:
        """Removes the table binding from this column.

        Returns
        -------
        ColumnReference
            The updated column reference, the original reference is not modified.
        """
        return ColumnReference(self.name, None)

    def __json__(self) -> object:
        return {"name": self._name, "table": self._table}

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ColumnReference):
            return NotImplemented
        if self.table == other.table:
            return self._normalized_name < other._normalized_name
        if not self.table:
            return True
        if not other.table:
            return False
        return self.table < other.table

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._normalized_name == other._normalized_name and self.table == other.table

    def __repr__(self) -> str:
        return f"ColumnReference(name='{self.name}', table={repr(self.table)})"

    def __str__(self) -> str:
        return self._sql_repr


class UnboundColumnError(StateError):
    """Indicates that a column is required to be bound to a table, but the provided column was not.

    Parameters
    ----------
    column : ColumnReference
        The column without the necessary table binding
    """

    def __init__(self, column: ColumnReference) -> None:
        super().__init__("Column is not bound to any table: " + str(column))
        self.column = column


class VirtualTableError(StateError):
    """Indicates that a table is required to correspond to a physical table, but the provided reference was not.

    Parameters
    ----------
    table : TableReference
        The virtual table
    """

    def __init__(self, table: TableReference) -> None:
        super().__init__("Table is virtual: " + str(table))
        self.table = table
