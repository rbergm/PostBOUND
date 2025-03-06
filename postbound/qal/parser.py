"""The parser constructs `SqlQuery` objects from query strings.

Other than the parsing itself, the process will also execute a basic column binding process. For example, consider
a query like *SELECT \\* FROM R WHERE R.a = 42*. In this case, the binding only affects the column reference *R.a*
and sets the table of that column to *R*. This binding based on column and table names is always performed.

If the table cannot be inferred based on the column name (e.g. for a query like *SELECT * FROM R, S WHERE a = 42*), a
second binding phase can be executed. This binding needs a working database connection and queries the database schema
to detect the correct tables for each column. Whether the second phase should also be executed by default can be
configured system-wide by setting the `auto_bind_columns` variable.

Notes
-----
Please beware that SQL parsing is a very challenging undertaking and there might be bugs in some lesser-used features.
If you encounter any issues, please report them on the GitHub issue tracker.
We test the parser based on some popular benchmarks, namely JOB and Stats to ensure that result sets from the raw SQL queries
match result sets from the parsed queries. However, we cannot guarantee that the parser will work for all SQL queries.

The parsing itself is based on the pglast project that implements a SQL -> JSON/dict conversion, based on the actual Postgres
query parser. Our parser implementation takes such a JSON representation as input and generates the more verbose structures of
the qal. There exists a Jupyter notebook called *PglastParsingTests* in the *tests* directory that shows the output emitted by
pglast for different SQL query features.

References
----------

.. pglast project: https://github.com/lelit/pglast Thanks a lot for maintaining this fantastic tool and the great support!
"""
from __future__ import annotations

import collections
import json
import warnings
from collections.abc import Iterable, Sequence
from typing import Literal, Optional, Protocol, overload

import pglast

from ._qal import (
    JoinType,
    CompoundOperator, MathOperator, LogicalOperator, SqlOperator, SetOperator,
    BaseProjection, OrderByExpression,
    WithQuery, TableSource, DirectTableSource, JoinTableSource, SubqueryTableSource,
    ValuesList, ValuesTableSource, ValuesWithQuery,
    SqlQuery, SetQuery, SelectStatement,
    Select, Where, From, GroupBy, Having, OrderBy, Limit, CommonTableExpression, ImplicitFromClause, ExplicitFromClause,
    Hint, Explain,
    AbstractPredicate, BinaryPredicate, InPredicate, BetweenPredicate, CompoundPredicate, UnaryPredicate,
    SqlExpression, StarExpression, StaticValueExpression, ColumnExpression, SubqueryExpression, CastExpression,
    MathExpression, FunctionExpression, WindowExpression, CaseExpression, ArrayAccessExpression,
    build_query
)
from .._core import TableReference, ColumnReference
from .. import util

auto_bind_columns: bool = True
"""Indicates whether the parser should use the database catalog to obtain column bindings."""


class DBCatalog(Protocol):
    """A simplified model of a database catalog that is used to resolve column bindings.

    See Also
    --------
    `DatabaseSchema` : The default implementation of a database catalog (we only distinguish between schema and catalog for
                       technical reasons to prevent circular imports).
    """

    def lookup_column(self, name: str | ColumnReference, candidates: Iterable[TableReference]) -> Optional[TableReference]:
        """Provides the table that defines a specific column.

        Returns
        -------
        Optional[TableReference]
            The table that defines the column. If there are multiple tables that could define the column, an arbitrary one
            is returned. If none of the candidates is the correct table, *None* is returned.
        """
        ...

    def columns(self, table: str) -> Sequence[ColumnReference]:
        """Provides the columns that belong to a specific table."""
        ...


class SchemaCache:
    """A simple cache that stores the columns that belong to tables in our database schema.

    The cache only queries the actual catalog of the database system, if the requested table has not been cached, yet.

    Parameters
    ----------
    schema : Optional[DatabaseSchema]
        The schema to cache. If not provided, the cache cannot resolve column bindings.
    """

    def __init__(self, schema: Optional[DBCatalog] = None) -> None:
        self._schema = schema
        self._lookup_cache: dict[TableReference, tuple[list[str], set[str]]] = collections.defaultdict(set)

    def initialize_with(self, schema: Optional[DBCatalog]) -> None:
        """Sets the catalog if necessary"""
        if self._schema is not None and self._schema != schema:
            warnings.warn("Parsing query for new schema. Dropping old schema cache.")
            self._schema = schema
            self._lookup_cache.clear()
        elif self._schema is not None:
            # same schema as before, do nothing
            return
        self._schema = schema

    def lookup_column(self, colname: str, candidate_tables: Iterable[TableReference]) -> Optional[TableReference]:
        """Resolves the table that defines a specific column.

        If no catalog is available, this method will always return *None*.

        Returns
        -------
        Optional[TableReference]
            The table that defines the column. If there are multiple tables that could define the column, an arbitrary one
            is returned. If none of the candidates is the correct table, *None* is returned.
        """
        if not self._schema:
            return None

        for candidate in candidate_tables:
            if candidate.virtual:
                continue

            _, table_columns = self._inflate_cache(candidate)
            if colname in table_columns:
                return candidate

        return None

    def columns_of(self, table: str) -> list[str]:
        """Provides the columns that belong to a specific table.

        If no catalog is available, this method will always return an empty list.
        """
        if not self._schema:
            return []

        cols, _ = self._inflate_cache(table)
        return cols

    def _inflate_cache(self, table: str) -> tuple[list[str], set[str]]:
        """Provides the columns that belong to a specific table, consulting the online catalog if necessary.

        This method assumes that there is indeed an online schema available. Calling this method without a schema will
        result in an arbitrary runtime error.

        Returns
        -------
        tuple[list[str], set[str]]
            The columns of the table in their defined order, as well as the same columns as a set.
        """
        cached_res = self._lookup_cache.get(table)
        if cached_res:
            return cached_res
        cols: list[str] = [col.name for col in self._schema.columns(table)]
        cols_set = set(cols)
        self._lookup_cache[table] = cols, cols_set
        return cols, cols_set


class QueryNamespace:
    """The query namespace acts as the central service to resolve column bindings in a query.

    It maintains a visibility map of all tables at a given point in the query and keeps track of the columns that form the
    result relation at the same points in time. This information is used to bind column references to the correct tables,
    including temporary virtual tables that alias existing physical columns.

    The namespace protocol works as follows:

    - While parsing a query, the table sources (CTEs and FROM entries) should be handled first. Each source should be
      registered in the namespace using the `register_table` method.
    - When a subquery or CTE is encoutered, the `open_nested` method has to be called to open a new local namespace and
      track the virtual table correctly.
    - Once all tables are registered, the parser can handle the *SELECT* clause. Afterwards, `determine_output_shape` has to
      called to compute all columns that are part of the result relation of the current namespace. This method takes care
      of resolving *SELECT \\** operations as necessary and requires that all input sources have already been registered and
      completely parsed, such that their output shapes are known.
    - While parsing the different clauses of the query, `lookup_column` and `resolve_table` can be used to determine the
      correct table references based on the sources that are currently available in the namespace.

    Each namespace can be connected to a parent namespace, which in turn can provide additional CTEs, physical tables or
    subqueries (if the current namespace is for a LATERAL subquery). This allows the current namespace to check whether some
    column is actually provided by an outer scope if the namespace does not provide the column itself.
    """

    _schema_cache: SchemaCache = SchemaCache()
    """The schema cache that is used to resolve column bindings. This cache is shared through the entire program lifetime.

    Changing the actual database schema while PostBOUND is running will result in undefined behavior.
    """

    @staticmethod
    def empty(schema: Optional["DatabaseSchema"] = None) -> QueryNamespace:  # type: ignore # noqa: F821
        QueryNamespace._schema_cache.initialize_with(schema)
        return QueryNamespace()

    def __init__(self, *, parent: Optional[QueryNamespace] = None) -> None:
        self._parent = parent

        self._subquery_children: dict[str, QueryNamespace] = {}
        """Nested namespaces that are provided as part of subqueries. Entries map alias -> query."""

        self._setop_children: list[QueryNamespace, QueryNamespace] = []
        """Namespace of the queries that form a set operation in the current namespace."""

        self._current_ctx: list[TableReference] = []
        """The tables that are currently in scope, no matter their origin (CTEs or FROM clause).

        For the purpose of this dictionary, it does not matter where a table comes from (physical table, CTE, subquery, ...).
        The only thing that matters is that the table is part of the FROM clause. This is especially important to build the
        correct output shape of the namespace/relation if the SELECT clause contains * expressions.

        Therefore, there might be tables that are contained in the `_cte_sources`, but not here (if the CTE is only used to
        build other CTEs, but not part of the FROM clause itself).

        Notice that tables that are defined in an enclosing scope (e.g. outer query in a sequence of nested CTEs) are not
        contained in this context if they are not also part of this namespace's *FROM* clause. Instead, they are resolved
        through the API on the parent namespace.

        The ordering is important to resolve output columns correctly to the first match as Postgres does.

        An optimized container to check whether a table is part of the current context is available via `_table_sources`
        """

        self._cte_sources: dict[str, QueryNamespace] = {}
        """Namespaces that are induced by CTEs. Entries map alias -> CTE."""

        self._table_sources: dict[str, TableReference] = {}
        """The tables that are part of the FROM clause of the query. Entries map alias -> table.

        Tables can be contained in this dictionary multiple times: once for each relevant identifier. If a table has both an
        alias as well as a full name, both keys will be present.
        """

        self._output_shape: list[str] = []
        """The column names that are part of the result set produced by the queries in this namespace.

        These are really just the column names, not full references. This is because it makes the access mechanism more
        transparent (just use the name, duh) and prevents accidental issues when a column from an inner query is re-used in an
        outer query and is bound to both virtual tables. Comparing the references would indicate that these are different
        columns (which arguably they are, just not for our purposes).
        """

        self._column_cache: dict[str, TableReference] = {}
        """A cache to resolve common columns in the current context more quickly."""

    def determine_output_shape(self, select_clause: Optional[Select | Iterable[ColumnReference | str]]) -> None:
        """Determines the columns that form the result relation of this namespace.

        The result is only stored internally to allow parent namespaces to resolve column references correctly.

        This method should only be called after all table sources from the current namespace are already registered in order to
        ensure that star expressions can be resolved correctly.
        """
        self._output_shape = []
        if self._setop_children:
            # We use Postgre's rules here: the output relation of a set operation contains exactly those columns that are
            # contained in the LHS relation
            self._output_shape = list(self._setop_children[0]._output_shape)
            return

        for projection in select_clause:
            if isinstance(projection, (str, ColumnReference)):
                self._output_shape.append(projection.name if isinstance(projection, ColumnReference) else projection)
                continue

            # must be BaseProjection
            if projection.target_name:
                self._output_shape.append(projection.target_name)
                continue

            match projection.expression:

                case ColumnExpression(col):
                    self._output_shape.append(col.name)

                case StarExpression(from_table):
                    ctx = {from_table} if from_table else self._current_ctx
                    for table in ctx:
                        if not table.virtual:
                            self._output_shape.extend(self._schema_cache.columns_of(table))
                            continue

                        defining_nsp = self._lookup_namespace(table.identifier())
                        if not defining_nsp:
                            continue
                        self._output_shape.extend(defining_nsp._output_shape)

                case _:
                    # do nothing, this is an expression that cannot be referenced later on!
                    pass

    def register_table(self, table: TableReference) -> None:
        """Adds a "physical" table to the current namespace.

        In truth, the table does not need to be physical, it can also be a CTE that was defined in an outer namespace and
        is scanned here. "Physical" in this context means that the current namespace does not define the table itself.
        """
        self._invalidate_column_cache()
        self._current_ctx.append(table)
        if table.alias:
            self._table_sources[table.alias] = table
        if table.full_name:
            self._table_sources[table.full_name] = table

    def provides_column(self, name: str) -> bool:
        """Checks, whether the current namespace has a specific column in its output relation."""
        return name in self._output_shape

    def lookup_column(self, key: str) -> Optional[TableReference]:
        """Searches for the table that provies a specific column.

        This table can be either virtual, i.e. a subquery or CTE (possibly from an outer namespace), or an actual physical
        table from the current database.

        If no table is found , *None* is returned.
        """
        cached_table = self._column_cache.get(key)
        if cached_table:
            return cached_table

        matching_table: Optional[TableReference] = None
        for table in self._current_ctx:
            # later tables overwrite unqualified columns of earlier tables
            physical_table = self._schema_cache.lookup_column(key, [table]) if not table.virtual else None
            if physical_table:
                matching_table = table
                break

            subquery_nsp = self._subquery_children.get(table.identifier())
            if subquery_nsp and subquery_nsp.provides_column(key):
                matching_table = table
                break

            cte_nsp = self._cte_sources.get(table.identifier())
            if cte_nsp and cte_nsp.provides_column(key):
                matching_table = table
                break

            parent_nsp = self._lookup_namespace(table.identifier())
            if parent_nsp and parent_nsp.provides_column(key):
                matching_table = table
                break

        if not matching_table:
            return None

        self._column_cache[key] = matching_table
        return matching_table

    def resolve_table(self, key: str) -> Optional[TableReference]:
        """Searches for the table that is referenced by a specific key.

        The table can be either provided by this namespace (as a physical table in the *FROM* clause, or defined through a
        subquery/CTE), or by an outer namspace.
        """
        sourced_table = self._table_sources.get(key)
        if sourced_table:
            return sourced_table

        if key in self._cte_sources:
            return TableReference.create_virtual(key)

        return self._parent.resolve_table(key) if self._parent else None

    def open_nested(self, *, alias: str = "",
                    source: Literal["cte", "subquery", "setop", "values", "temporary"]) -> QueryNamespace:
        """Creates a new local namespace for a nested query.

        Depending on the type of nested query, the namespace will be registered in different ways and used for different
        purposes (see parameters below).

        Parameters
        ----------
        alias : str, optional
            The name of the namespace. This is only relevant for CTEs and subqueries in the FROM clause.
        source : Literal["cte", "subquery", "setop", "values", "temporary"]
            The type of nested query. This value is used to determine the use of the subquery namespace as follows:
            - "cte": The namespace is a CTE that is part of the query.
            - "subquery": The namespace is a subquery in the FROM clause.
            - "setop": The namespace is part of a set operation. No alias is required, but the namespace might be used to
              determine the output shape of the current namespace
            - "values": The namespace is a temporary table that is part of a VALUES clause.
            - "temporary": The namespace is a temporary table that is part of a subquery which is not used in the FROM clause,
              e.g. as a filter condition.
        """
        if source != "temporary":
            self._invalidate_column_cache()

        child = QueryNamespace(parent=self)

        match source:
            case "cte":
                self._cte_sources[alias] = child
            case "subquery" | "values":
                table = TableReference.create_virtual(alias)
                self._subquery_children[alias] = child
                self._current_ctx.append(table)
                self._table_sources[alias] = table
            case "setop":
                self._setop_children.append(child)
            case _:
                # ignore other sources
                pass

        return child

    def _lookup_namespace(self, table_key: str) -> Optional[QueryNamespace]:
        """Searches for the (parent) namespace that provides a specific table."""
        cte_nsp = self._cte_sources.get(table_key)
        if cte_nsp:
            return cte_nsp

        subquery_nsp = self._subquery_children.get(table_key)
        if subquery_nsp:
            return subquery_nsp

        return self._parent._lookup_namespace(table_key) if self._parent else None

    def _invalidate_column_cache(self) -> None:
        """Clears all currently cached columns in case there is fear of a change in the column bindings."""
        self._column_cache.clear()


def _pglast_is_actual_colref(pglast_data: dict) -> bool:
    """Checks, whether a apparent column reference is actually a column reference and not a star expression in disguise.

    pglast represents both column references such as *R.a* or *a* as well as star expressions like *R.\\** as ``ColumnRef``
    dictionaries, hence we need to make sure we are actually parsing the right thing. This method takes care of distinguishing
    the two cases.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the potential column

    Returns
    -------
    bool
        *True* if this is an actual column reference, *False* if this is a star expression.
    """
    fields: list[dict] = pglast_data["fields"]
    if len(fields) == 1:
        return "A_Star" not in fields[0]
    if len(fields) == 2:
        would_be_col: str = fields[1]
        return "A_Star" not in would_be_col

    would_be_col: str = fields[0]["String"]["sval"]
    return not would_be_col.endswith("*")


def _pglast_create_bound_colref(tab: str, col: str, *, namespace: QueryNamespace) -> ColumnReference:
    """Creates a new reference to a column with known binding info.

    Parameters
    ----------
    tab : str
        The table to which to bind
    col : str
        The column to bind
    namespace : QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    ColumnReference
        The new column reference
    """
    owning_table = namespace.resolve_table(tab)
    if not owning_table:
        raise ParserError("Table not found: " + tab)
    parsed_column = ColumnReference(col, owning_table)
    return parsed_column


def _pglast_parse_colref(pglast_data: dict, *, namespace: QueryNamespace) -> ColumnReference:
    """Handler method to parse column references in the query.

    The column will be bound to its table if possible. This binding process uses the following rules:

    - if the columns has already been resolved as part of an earlier parsing step in the same namespace, this column is re-used
    - if the column is specified in qualified syntax (i.e. *table.column*), the table is directly inferred
    - if the column is not qualified, but a `schema` is given, this schema is used together with the candidates from the
      current namespace to lookup the owning table
    - otherwise, the column is left unbound :(

    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the column
    namespace : QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    ColumnReference
        The parsed column reference.
    """
    fields: list[dict] = pglast_data["fields"]
    if len(fields) > 2:
        raise ParserError("Unknown column reference format: " + str(pglast_data))

    if len(fields) == 2:
        tab, col = fields
        return _pglast_create_bound_colref(tab["String"]["sval"], col["String"]["sval"], namespace=namespace)

    # at this point, we must have a single column parameter. It could be unbounded, or - if quoted - bounded
    col: str = fields[0]["String"]["sval"]
    owning_table = namespace.lookup_column(col)
    return ColumnReference(col, owning_table)


def _pglast_parse_star(pglast_data: dict, *, namespace: QueryNamespace) -> StarExpression:
    """Handler method to parse star expressions that are potentially bounded to a specific table, e.g. *R.\\**.

    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the star expression
    namespace : QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    StarExpression
        The parsed star expression.
    """
    fields = pglast_data["fields"]
    if len(fields) == 1 and "A_Star" in fields[0]:
        return StarExpression()

    if len(fields) == 2:
        tab = fields[0]["String"]["sval"]
        return StarExpression(from_table=namespace.resolve_table(tab))

    raise ParserError("Unknown star reference format: " + str(pglast_data))


def _pglast_parse_const(pglast_data: dict) -> StaticValueExpression:
    """Handler method to parse constant values in the query.

    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the value. This data is extracted from the pglast data structure.

    Returns
    -------
    StaticValueExpression
        The parsed constant value.
    """
    pglast_data.pop("location", None)
    valtype = util.dicts.key(pglast_data)
    match valtype:
        case "isnull":
            return StaticValueExpression.null()
        case "ival":
            val = pglast_data["ival"]["ival"] if "ival" in pglast_data["ival"] else 0
            return StaticValueExpression(val)
        case "fval":
            val = pglast_data["fval"]["fval"]
            return StaticValueExpression(float(val))
        case "sval":
            return StaticValueExpression(pglast_data["sval"]["sval"])
        case "boolval":
            val = pglast_data["boolval"].get("boolval", False)
            return StaticValueExpression(val)
        case _:
            raise ParserError("Unknown constant type: " + str(pglast_data))


_PglastOperatorMap: dict[str, SqlOperator] = {
    "=": LogicalOperator.Equal,
    "<": LogicalOperator.Less,
    "<=": LogicalOperator.LessEqual,
    ">": LogicalOperator.Greater,
    ">=": LogicalOperator.GreaterEqual,
    "<>": LogicalOperator.NotEqual,
    "!=": LogicalOperator.NotEqual,

    "AND_EXPR": CompoundOperator.And,
    "OR_EXPR": CompoundOperator.Or,
    "NOT_EXPR": CompoundOperator.Not,

    "+": MathOperator.Add,
    "-": MathOperator.Subtract,
    "*": MathOperator.Multiply,
    "/": MathOperator.Divide,
    "%": MathOperator.Modulo,
    "||": MathOperator.Concatenate,
}
"""Map from the internal representation of Postgres operators to our standardized QAL operators."""


def _pglast_parse_operator(pglast_data: list[dict]) -> SqlOperator:
    """Handler method to parse operators into our query representation.

    Parameters
    ----------
    pglast_data : list[dict]
        JSON enconding of the operator. This data is extracted from the pglast data structure.

    Returns
    -------
    SqlOperator
        The parsed operator.
    """
    if len(pglast_data) != 1:
        raise ParserError("Unknown operator format: " + str(pglast_data))
    operator = pglast_data[0]
    if "String" not in operator or "sval" not in operator["String"]:
        raise ParserError("Unknown operator format: " + str(pglast_data))
    sval = operator["String"]["sval"]
    if sval not in _PglastOperatorMap:
        raise ParserError("Operator not yet in target map: " + sval)
    return _PglastOperatorMap[sval]


_PglastTypeMap: dict[str, str] = {
    "bpchar": "char",

    "serial8": "bigserial",


    "int4": "integer",
    "int2": "smallint",
    "int8": "bigint",

    "float4": "real",
    "float8": "double precision",

    "boolean": "bool"
}
"""Map from the internal representation of Postgres types to the SQL standard types."""


def _pglast_parse_type(pglast_data: dict) -> str:
    """Handler method to parse type information from explicit type casts

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the type information.

    Returns
    -------
    str
        The actual type
    """
    if "names" not in pglast_data:
        raise ParserError("Unknown type format: " + str(pglast_data))
    names = pglast_data["names"]
    if len(names) > 2:
        raise ParserError("Unknown type format: " + str(pglast_data))
    raw_type = names[-1]["String"]["sval"]

    # for user-defined types we use get with the same type as argument
    return _PglastTypeMap.get(raw_type, raw_type)


def _pglast_parse_case(pglast_data: dict, *, namespace: QueryNamespace) -> CaseExpression:
    """Handler method to parse *CASE* expressions in a query.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the *CASE* expression data. This data is extracted from the pglast data structure.
    namespace : QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    CaseExpression
        The parsed *CASE* expression.
    """
    cases: list[tuple[AbstractPredicate, SqlExpression]] = []
    for arg in pglast_data["args"]:
        current_case = _pglast_parse_predicate(arg["CaseWhen"]["expr"], namespace=namespace)
        current_result = _pglast_parse_expression(arg["CaseWhen"]["result"], namespace=namespace)
        cases.append((current_case, current_result))

    if "defresult" in pglast_data:
        default_result = _pglast_parse_expression(pglast_data["defresult"], namespace=namespace)
    else:
        default_result = None

    return CaseExpression(cases, else_expr=default_result)


def _pglast_parse_expression(pglast_data: dict, *, namespace: QueryNamespace) -> SqlExpression:
    """Handler method to parse arbitrary expressions in the query.

    For some more complex expressions, this method will delegate to tailored parsing methods.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the expression data. This data is extracted from the pglast data structure.
    namespace: QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    SqlExpression
        The parsed expression.
    """
    pglast_data.pop("location", None)
    expression_key = util.dicts.key(pglast_data)

    # When parsing the actual expression, we need to be aware that many expressions can actually be predicates, just not
    # within the WHERE or HAVING clause. For example, "SELECT a IS NOT NULL FROM foo" is a perfectly valid query.
    # Therefore, we handle a lot of expression cases by passing the input data back to our predicate parser and let it do the
    # heavy lifting.

    match expression_key:

        case "ColumnRef" if _pglast_is_actual_colref(pglast_data["ColumnRef"]):
            column = _pglast_parse_colref(pglast_data["ColumnRef"], namespace=namespace)
            return ColumnExpression(column)

        case "ColumnRef" if not _pglast_is_actual_colref(pglast_data["ColumnRef"]):
            return _pglast_parse_star(pglast_data["ColumnRef"], namespace=namespace)

        case "A_Const":
            return _pglast_parse_const(pglast_data["A_Const"])

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_OP":
            expression = pglast_data["A_Expr"]
            operation = _pglast_parse_operator(expression["name"])
            right = _pglast_parse_expression(expression["rexpr"], namespace=namespace)

            if "lexpr" not in expression and operation in MathOperator:
                return MathExpression(operation, right)
            elif "lexpr" not in expression:
                raise ParserError("Unknown operator format: " + str(expression))

            left = _pglast_parse_expression(expression["lexpr"], namespace=namespace)

            if operation in LogicalOperator:
                return BinaryPredicate(operation, left, right)

            return MathExpression(operation, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] in {"AEXPR_LIKE", "AEXPR_ILIKE", "AEXPR_BETWEEN", "AEXPR_IN"}:
            # we need to parse a predicate in disguise
            predicate = _pglast_parse_predicate(pglast_data, namespace=namespace)
            return predicate

        case "NullTest":
            predicate = _pglast_parse_predicate(pglast_data, namespace=namespace)
            return predicate

        case "BoolExpr":
            predicate = _pglast_parse_predicate(pglast_data, namespace=namespace)
            return predicate

        case "FuncCall" if "over" not in pglast_data["FuncCall"]:  # normal functions, aggregates and UDFs
            expression: dict = pglast_data["FuncCall"]
            funcname = ".".join(elem["String"]["sval"] for elem in expression["funcname"])
            distinct = expression.get("agg_distinct", False)
            if expression.get("agg_filter", False):
                filter_expr = _pglast_parse_predicate(expression["agg_filter"], namespace=namespace)
            else:
                filter_expr = None

            if expression.get("agg_star", False):
                return FunctionExpression(funcname, [StarExpression()], distinct=distinct, filter_where=filter_expr)

            args = [_pglast_parse_expression(arg, namespace=namespace) for arg in expression.get("args", [])]
            return FunctionExpression(funcname, args, distinct=distinct, filter_where=filter_expr)

        case "FuncCall" if "over" in pglast_data["FuncCall"]:  # window functions
            expression: dict = pglast_data["FuncCall"]
            funcname = ".".join(elem["String"]["sval"] for elem in expression["funcname"])

            args = [_pglast_parse_expression(arg, namespace=namespace) for arg in expression.get("args", [])]
            fn = FunctionExpression(funcname, args)

            window_spec: dict = expression["over"]

            if "partitionClause" in window_spec:
                partition = [_pglast_parse_expression(partition, namespace=namespace)
                             for partition in window_spec["partitionClause"]]
            else:
                partition = None

            if "orderClause" in window_spec:
                order = _pglast_parse_orderby(window_spec["orderClause"], namespace=namespace)
            else:
                order = None

            if "agg_filter" in expression:
                filter_expr = _pglast_parse_expression(expression["agg_filter"], namespace=namespace)
            else:
                filter_expr = None

            return WindowExpression(fn, partitioning=partition, ordering=order, filter_condition=filter_expr)

        case "CoalesceExpr":
            expression = pglast_data["CoalesceExpr"]
            args = [_pglast_parse_expression(arg, namespace=namespace) for arg in expression["args"]]
            return FunctionExpression("coalesce", args)

        case "TypeCast":
            expression: dict = pglast_data["TypeCast"]
            casted_expression = _pglast_parse_expression(expression["arg"], namespace=namespace)
            target_type = _pglast_parse_type(expression["typeName"])
            type_params = [_pglast_parse_expression(param, namespace=namespace)
                           for param in expression["typeName"].get("typmods", [])]

            return CastExpression(casted_expression, target_type, type_params=type_params)

        case "CaseExpr":
            return _pglast_parse_case(pglast_data["CaseExpr"], namespace=namespace)

        case "SubLink" if pglast_data["SubLink"]["subLinkType"] == "EXPR_SUBLINK":
            subquery = _pglast_parse_query(pglast_data["SubLink"]["subselect"]["SelectStmt"],
                                           namespace=namespace.open_nested(source="temporary"))
            return SubqueryExpression(subquery)

        case "A_Indirection":
            expression: dict = pglast_data["A_Indirection"]
            array_expression = _pglast_parse_expression(expression["arg"], namespace=namespace)

            for index_expression in expression["indirection"]:
                index_expression: dict = index_expression["A_Indices"]

                if index_expression.get("is_slice", False):
                    lower = (_pglast_parse_expression(index_expression["lidx"], namespace=namespace)
                             if "lidx" in index_expression else None)
                    upper = (_pglast_parse_expression(index_expression["uidx"], namespace=namespace)
                             if "uidx" in index_expression else None)
                    array_expression = ArrayAccessExpression(array_expression, lower_idx=lower, upper_idx=upper)
                    continue

                point_index = _pglast_parse_expression(index_expression["uidx"], namespace=namespace)
                array_expression = ArrayAccessExpression(array_expression, idx=point_index)

            return array_expression

        case _:
            raise ParserError("Unknown expression type: " + str(pglast_data))


def _pglast_parse_values_cte(pglast_data: dict, *, namespace: QueryNamespace) -> tuple[ValuesList, list[str]]:
    """Handler method to parse a CTE with a *VALUES* expressions.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the CTE data. This data is extracted from the pglast data structure.
    namespace: QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    tuple[ValuesList, list[str]]
        The parsed *VALUES* expression and the column names.
    """
    values: ValuesList = []
    for row in pglast_data["ctequery"]["SelectStmt"]["valuesLists"]:
        raw_items = row["List"]["items"]
        parsed_items = [_pglast_parse_expression(item, namespace=namespace)
                        for item in raw_items]
        values.append(tuple(parsed_items))

    colnames: list[str] = []
    for raw_colname in pglast_data.get("aliascolnames", []):
        colnames.append(raw_colname["String"]["sval"])

    if colnames:
        namespace.determine_output_shape(colnames)

    return values, colnames


def _pglast_parse_ctes(json_data: dict, *, parent_namespace: QueryNamespace) -> CommonTableExpression:
    """Handler method to parse the *WITH* clause of a query.

    Parameters
    ----------
    json_data : dict
        JSON enconding of the CTEs, as extracted from the pglast data structure.
    parent_namespace: QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    CommonTableExpression
        The parsed CTEs.
    """
    parsed_ctes: list[CommonTableExpression] = []
    for pglast_data in json_data["ctes"]:
        current_cte: dict = pglast_data["CommonTableExpr"]
        target_name = current_cte["ctename"]
        target_table = TableReference.create_virtual(target_name)

        match current_cte.get("ctematerialized", "CTEMaterializeDefault"):
            case "CTEMaterializeDefault":
                force_materialization = None
            case "CTEMaterializeAlways":
                force_materialization = True
            case "CTEMaterializeNever":
                force_materialization = False

        query_data = current_cte["ctequery"]["SelectStmt"]
        child_nsp = parent_namespace.open_nested(alias=target_name, source="cte")
        if "targetList" not in query_data and query_data["op"] == "SETOP_NONE":
            # CTE is a VALUES query
            values, columns = _pglast_parse_values_cte(current_cte, namespace=child_nsp)
            parsed_cte = ValuesWithQuery(values, target_name=target_table.identifier(),
                                         columns=columns, materialized=force_materialization)
        else:
            cte_query = _pglast_parse_query(current_cte["ctequery"]["SelectStmt"], namespace=child_nsp)
            parsed_cte = WithQuery(cte_query, target_table, materialized=force_materialization)

        parsed_ctes.append(parsed_cte)

    recursive = json_data.get("recursive", False)
    return CommonTableExpression(parsed_ctes, recursive=recursive)


def _pglast_try_select_star(target: dict, *, distinct: list[SqlExpression] | bool) -> Optional[Select]:
    """Attempts to generate a *SELECT(\\*)* representation for a *SELECT* clause.

    If the query is not actually a *SELECT(\\*)* query, this method will return *None*.

    Parameters
    ----------
    target : dict
        JSON encoding of the target entry in the *SELECT* clause. This data is extracted from the pglast data structure
    distinct : list[SqlExpression] | bool
        The parsed *DISTINCT* part of the *SELECT* clause.

    Returns
    -------
    Optional[Select]
        The parsed *SELECT(\\*)* clause, or *None* if this is not a *SELECT(\\*)* query.
    """
    if "ColumnRef" not in target:
        return None
    fields = target["ColumnRef"]["fields"]
    if len(fields) != 1:
        # multiple fields are used for qualified column references. This is definitely not a SELECT * query, so exit
        return None
    colref = fields[0]
    return Select.star(distinct=distinct) if "A_Star" in colref else None


def _pglast_parse_select(pglast_data: dict, *, namespace: QueryNamespace) -> Select:
    """Handler method to parse the *SELECT* clause of a query.

    This is the only parsing handler that will always be called when parsing a query, since all queries must at least have a
    *SELECT* clause.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the entire query. This is required to extract the different projections used in the *SELECT* clause,
        as well as potential required duplicate eliminations via *DISTINCT ON*
    namespace : QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    Select
        The parsed *SELECT* clause
    """

    pglast_distinct = pglast_data.get("distinctClause", None)
    if pglast_distinct is None:
        distinct = False  # value not present --> no DISTINCT
    elif pglast_distinct == [{}]:  # that is pglasts encoding of a plain DISTINCT
        distinct = True
    elif isinstance(pglast_distinct, list):
        distinct = [_pglast_parse_expression(expr, namespace=namespace) for expr in pglast_distinct]
    else:
        raise ParserError(f"Unknown DISTINCT format: {pglast_distinct}")

    targetlist: list[dict] = pglast_data["targetList"]
    # first, try for SELECT * queries
    if len(targetlist) == 1:
        target = targetlist[0]["ResTarget"]["val"]
        select_star = _pglast_try_select_star(target, distinct=distinct)

        if select_star:
            return select_star
        # if this is not a SELECT * query, we can continue with the regular parsing

    targets: list[BaseProjection] = []
    for target in targetlist:
        expression = _pglast_parse_expression(target["ResTarget"]["val"], namespace=namespace)
        alias = target["ResTarget"].get("name", "")
        projection = BaseProjection(expression, alias)
        targets.append(projection)

    clause = Select(targets, distinct=distinct)
    namespace.determine_output_shape(clause)
    return clause


def _pglast_parse_rangevar(rangevar: dict) -> TableReference:
    """Handler method to extract the `TableReference` from a *RangeVar* entry in the *FROM* clause.

    Parameters
    ----------
    rangevar : dict
        JSON encoding of the range variable, as extracted from the pglast data structure.

    Returns
    -------
    TableReference
        The parsed table reference.
    """
    name = rangevar["relname"]
    alias = rangevar["alias"]["aliasname"] if "alias" in rangevar else None
    schema = rangevar.get("schemaname", "")
    return TableReference(name, alias, schema=schema)


def _pglast_is_values_list(pglast_data: dict) -> bool:
    """Checks, whether a pglast subquery representation refers to an actual subquery or a *VALUES* list.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the subquery data

    Returns
    -------
    bool
        *True* if the subquery encodes a *VALUES* list, *False* otherwise.
    """
    query = pglast_data["subquery"]["SelectStmt"]
    return "valuesLists" in query


def _pglast_parse_from_entry(pglast_data: dict, *, namespace: QueryNamespace) -> TableSource:
    """Handler method to parse individual entries in the *FROM* clause.

    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the current entry in the *FROM* clause. This data is extracted from the pglast data structure.
    namespace: QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    TableSource
        The parsed table source.
    """
    pglast_data.pop("location", None)
    entry_type = util.dicts.key(pglast_data)

    match entry_type:

        case "RangeVar":
            table = _pglast_parse_rangevar(pglast_data["RangeVar"])

            # If we specified a virtual table in a CTE, we will reference it later in some FROM clause. In this case,
            # we should not create a new table reference, but rather use the existing one.
            # But, if we alias the virtual table, we still need a new reference
            similar_table = namespace.resolve_table(table.full_name)
            if similar_table and similar_table.virtual and not table.alias:
                # a simple reference to the CTE
                namespace.register_table(similar_table)
                return DirectTableSource(similar_table)
            if similar_table and similar_table.virtual and table.alias:
                # an aliased reference to the CTE
                table = table.make_virtual()
                # TODO: should we also update the mapping of the full_name here?

            namespace.register_table(table)
            return DirectTableSource(table)

        case "JoinExpr":
            join_expr: dict = pglast_data["JoinExpr"]
            match join_expr["jointype"]:
                case "JOIN_INNER" if "quals" in join_expr:
                    join_type = JoinType.InnerJoin
                case "JOIN_INNER" if "quals" not in join_expr:
                    join_type = JoinType.CrossJoin
                case "JOIN_LEFT":
                    join_type = JoinType.LeftJoin
                case "JOIN_RIGHT":
                    join_type = JoinType.RightJoin
                case "JOIN_OUTER":
                    join_type = JoinType.OuterJoin
                case "JOIN_FULL":
                    join_type = JoinType.OuterJoin
                case _:
                    raise ParserError("Unknown join type: " + join_expr["jointype"])

            left = _pglast_parse_from_entry(join_expr["larg"], namespace=namespace)
            right = _pglast_parse_from_entry(join_expr["rarg"], namespace=namespace)
            if join_type == JoinType.CrossJoin:
                return JoinTableSource(left, right, join_type=JoinType.CrossJoin)

            join_condition = _pglast_parse_predicate(join_expr["quals"], namespace=namespace)

            # we do not need to store new tables in available_tables here, since this is already handled by the recursion.
            return JoinTableSource(left, right, join_condition=join_condition, join_type=join_type)

        case "RangeSubselect" if _pglast_is_values_list(pglast_data["RangeSubselect"]):
            values_list = _pglast_parse_values(pglast_data["RangeSubselect"], parent_namespace=namespace)
            return values_list

        case "RangeSubselect":
            raw_subquery: dict = pglast_data["RangeSubselect"]
            is_lateral = raw_subquery.get("lateral", False)

            if "alias" in raw_subquery:
                alias: str = raw_subquery["alias"]["aliasname"]
            else:
                alias = ""

            child_nsp = namespace.open_nested(alias=alias, source="subquery")
            subquery = _pglast_parse_query(raw_subquery["subquery"]["SelectStmt"], namespace=child_nsp)

            subquery_source = SubqueryTableSource(subquery, target_name=alias, lateral=is_lateral)
            return subquery_source

        case _:
            raise ParserError("Unknow FROM clause entry: " + str(pglast_data))


def _pglast_parse_values(pglast_data: dict, *, parent_namespace: QueryNamespace) -> ValuesTableSource:
    """Handler method to parse explicit *VALUES* lists in the *FROM* clause.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the actual *VALUES* list. This data is extracted from the pglast data structure and should be akin
        to a subquery.
    parent_namespace : QueryNamespace
        The tables and columns that are available in the current query. This is only used to register the columns of the
        VALUES list

    Returns
    -------
    ValuesTableSource
        The parsed *VALUES* list.
    """

    raw_alias: dict = pglast_data.get("alias", {})
    alias = raw_alias.get("aliasname", "")
    child_nsp = parent_namespace.open_nested(alias=alias, source="values")
    raw_values: list[dict] = pglast_data["subquery"]["SelectStmt"]["valuesLists"]

    values: ValuesList = []
    for row in raw_values:
        raw_items = row["List"]["items"]
        parsed_items = [_pglast_parse_expression(item, namespace=child_nsp)
                        for item in raw_items]
        values.append(tuple(parsed_items))

    if not alias:
        return ValuesTableSource(values, alias=alias, columns=[])

    if "colnames" not in raw_alias:
        return ValuesTableSource(values, alias=alias, columns=[])

    colnames = []
    for raw_colname in raw_alias["colnames"]:
        colnames.append(raw_colname["String"]["sval"])
    table_source = ValuesTableSource(values, alias=alias, columns=colnames)
    child_nsp.determine_output_shape(table_source.cols)
    return table_source


def _pglast_parse_from(from_clause: list[dict], *, namespace: QueryNamespace) -> From:
    """Handler method to parse the *FROM* clause of a query.

    Parameters
    ----------
    from_clause : list[dict]
        The JSON representation of the *FROM* clause, as extracted from the pglast data structure.
    namespace : QueryNamespace
        The tables and columns that are available in the current query.

    Returns
    -------
    From
        The parsed *FROM* clause.
    """
    contains_plain_table = False
    contains_join = False
    contains_mixed = False  # plain tables and explicit JOINs, subqueries or VALUES

    table_sources: list[TableSource] = []
    for entry in from_clause:
        current_table_source = _pglast_parse_from_entry(entry, namespace=namespace)
        table_sources.append(current_table_source)

        match current_table_source:
            case DirectTableSource():
                contains_plain_table = True
                if contains_join:
                    contains_mixed = True
            case JoinTableSource():
                contains_join = True
                if contains_plain_table:
                    contains_mixed = True
            case SubqueryTableSource():
                contains_mixed = True
            case ValuesTableSource():
                contains_mixed = True

    if not contains_join and not contains_mixed:
        return ImplicitFromClause(table_sources)
    if contains_join and not contains_mixed:
        return ExplicitFromClause(table_sources)

    return From(table_sources)


def _pglast_parse_predicate(pglast_data: dict, *, namespace: QueryNamespace) -> AbstractPredicate:
    """Handler method to parse arbitrary predicates in the *WHERE* or *HAVING* clause.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the predicate data. This data is extracted from the pglast data structure.
    namespace : QueryNamespace
        The tables and columns that are available in the current query.


    Returns
    -------
    AbstractPredicate
        The parsed predicate.
    """
    pglast_data.pop("location", None)
    expr_key = util.dicts.key(pglast_data)
    match expr_key:

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_OP":
            expression = pglast_data["A_Expr"]
            operator = _pglast_parse_operator(expression["name"])
            left = _pglast_parse_expression(expression["lexpr"], namespace=namespace)
            right = _pglast_parse_expression(expression["rexpr"], namespace=namespace)
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_LIKE":
            expression = pglast_data["A_Expr"]
            operator = (LogicalOperator.Like if expression["name"][0]["String"]["sval"] == "~~"
                        else LogicalOperator.NotLike)
            left = _pglast_parse_expression(expression["lexpr"], namespace=namespace)
            right = _pglast_parse_expression(expression["rexpr"], namespace=namespace)
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_ILIKE":
            expression = pglast_data["A_Expr"]
            operator = (LogicalOperator.ILike if expression["name"][0]["String"]["sval"] == "~~*"
                        else LogicalOperator.NotILike)
            left = _pglast_parse_expression(expression["lexpr"], namespace=namespace)
            right = _pglast_parse_expression(expression["rexpr"], namespace=namespace)
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_BETWEEN":
            expression = pglast_data["A_Expr"]
            left = _pglast_parse_expression(expression["lexpr"], namespace=namespace)
            raw_interval = expression["rexpr"]["List"]["items"]
            if len(raw_interval) != 2:
                raise ParserError("Invalid BETWEEN interval: " + str(raw_interval))
            lower = _pglast_parse_expression(raw_interval[0], namespace=namespace)
            upper = _pglast_parse_expression(raw_interval[1], namespace=namespace)
            return BetweenPredicate(left, (lower, upper))

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_IN":
            expression = pglast_data["A_Expr"]
            left = _pglast_parse_expression(expression["lexpr"], namespace=namespace)
            raw_values = expression["rexpr"]["List"]["items"]
            values = [_pglast_parse_expression(value, namespace=namespace)
                      for value in raw_values]
            predicate = InPredicate(left, values)
            operator = expression["name"][0]["String"]["sval"]
            if operator == "=":
                return predicate
            elif operator == "<>":
                return CompoundPredicate.create_not(predicate)
            else:
                raise ParserError("Invalid IN operator: " + operator)

        case "BoolExpr":
            expression = pglast_data["BoolExpr"]
            operator = _PglastOperatorMap[expression["boolop"]]
            children = [_pglast_parse_predicate(child, namespace=namespace) for child in expression["args"]]
            return CompoundPredicate(operator, children)

        case "NullTest":
            expression = pglast_data["NullTest"]
            testexpr = _pglast_parse_expression(expression["arg"], namespace=namespace)
            operation = LogicalOperator.Is if expression["nulltesttype"] == "IS_NULL" else LogicalOperator.IsNot
            return BinaryPredicate(operation, testexpr, StaticValueExpression.null())

        case "FuncCall":
            expression = _pglast_parse_expression(pglast_data, namespace=namespace)
            return UnaryPredicate(expression)

        case "SubLink":
            expression = pglast_data["SubLink"]
            sublink_type = expression["subLinkType"]

            subquery = _pglast_parse_query(expression["subselect"]["SelectStmt"], namespace=namespace)
            if sublink_type == "EXISTS_SUBLINK":
                return UnaryPredicate.exists(subquery)

            testexpr = _pglast_parse_expression(expression["testexpr"], namespace=namespace)

            if sublink_type == "ANY_SUBLINK" and "operName" not in expression:
                return InPredicate.subquery(testexpr, subquery)

            if sublink_type == "ANY_SUBLINK":
                operator = _PglastOperatorMap[expression["operName"]]
                subquery_expression = FunctionExpression.any_func(subquery)
                return BinaryPredicate(operator, testexpr, subquery_expression)
            elif sublink_type == "ALL_SUBLINK":
                operator = _PglastOperatorMap[expression["operName"]]
                subquery_expression = FunctionExpression.all_func(subquery)
                return BinaryPredicate(operator, testexpr, subquery_expression)
            else:
                raise NotImplementedError("Subquery handling is not yet implemented")

        case _:
            expression = _pglast_parse_expression(pglast_data, namespace=namespace)
            return UnaryPredicate(expression)


def _pglast_parse_where(where_clause: dict, *, namespace: QueryNamespace) -> Where:
    """Handler method to parse the *WHERE* clause of a query.

    Parameters
    ----------
    where_clause : dict
        The JSON representation of the *WHERE* clause, as extracted from the pglast data structure.
    namespace: QueryNamespace
        The tables and columns that can be referenced by expressions in the query.

    Returns
    -------
    Where
        The parsed *WHERE* clause.
    """
    predicate = _pglast_parse_predicate(where_clause, namespace=namespace)
    return Where(predicate)


def _pglast_parse_groupby(groupby_clause: list[dict], *, namespace: QueryNamespace) -> GroupBy:
    """Handler method to parse the *GROUP BY* clause of a query.

    Parameters
    ----------
    groupby_clause : list[dict]
        The JSON representation of the *GROUP BY* clause, as extracted from the pglast data structure
    namespace: QueryNamespace
        The tables and columns that can be referenced by expressions in the query

    Returns
    -------
    GroupBy
        The parsed *GROUP BY* clause.
    """
    groupings: list[SqlExpression] = []

    for item in groupby_clause:
        if "GroupingSet" in item:
            raise NotImplementedError("Grouping sets are not yet supported")
        group_expression = _pglast_parse_expression(item, namespace=namespace)
        groupings.append(group_expression)

    return GroupBy(groupings)


def _pglast_parse_having(having_clause: dict, *, namespace: QueryNamespace) -> Having:
    """Handler method to parse the *HAVING* clause of a query.

    Parameters
    ----------
    having_clause : dict
        The JSON representation of the *HAVING* clause, as extracted from the pglast data structure.
    namespace: QueryNamespace
        The tables and columns that can be referenced by expressions in the query.

    Returns
    -------
    Having
        The parsed *HAVING* clause.
    """
    predicate = _pglast_parse_predicate(having_clause, namespace=namespace)
    return Having(predicate)


def _pglast_parse_orderby(order_clause: list[dict], *, namespace: QueryNamespace) -> OrderBy:
    """Handler method to parse the *ORDER BY* clause of a query.

    Parameters
    ----------
    order_clause : list[dict]
        The JSON representation of the *ORDER BY* clause, as extracted from the pglast data structure.
    namespace : QueryNamespace
        The tables and columns that can be referenced by expressions in the query.

    Returns
    -------
    OrderBy
        The parsed *ORDER BY* clause.
    """
    orderings: list[OrderByExpression] = []

    for item in order_clause:
        expression = item["SortBy"]
        sort_key = _pglast_parse_expression(expression["node"], namespace=namespace)

        match expression["sortby_dir"]:
            case "SORTBY_ASC":
                sort_ascending = True
            case "SORTBY_DESC":
                sort_ascending = False
            case "SORTBY_DEFAULT":
                sort_ascending = None
            case _:
                raise ParserError("Unknown sort direction: " + expression["sortby_dir"])

        match expression["sortby_nulls"]:
            case "SORTBY_NULLS_FIRST":
                put_nulls_first = True
            case "SORTBY_NULLS_LAST":
                put_nulls_first = False
            case "SORTBY_NULLS_DEFAULT":
                put_nulls_first = None
            case _:
                raise ParserError("Unknown nulls placement: " + expression["sortby_nulls"])

        order_expression = OrderByExpression(sort_key, ascending=sort_ascending, nulls_first=put_nulls_first)
        orderings.append(order_expression)

    return OrderBy(orderings)


def _pglast_parse_limit(pglast_data: dict, *, namespace: QueryNamespace) -> Optional[Limit]:
    """Handler method to parse LIMIT and OFFSET clauses.

    This method assumes that the given query actually contains *LIMIT* or *OFFSET* clauses and will fail otherwise.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the entire query. The method takes care of accessing the appropriate keys by itself, no preparation
        of the ``SelectStmt`` is necessary.
    namespace : QueryNamespace
        The tables and columns that can be referenced by expressions in the query.

    Returns
    -------
    Limit
        The limit clause. Can be *None* if no meaningful limit nor a meaningful offset is specified.
    """
    raw_limit: Optional[dict] = pglast_data.get("limitCount", None)
    raw_offset: Optional[dict] = pglast_data.get("limitOffset", None)
    if raw_limit is None and raw_offset is None:
        return None

    if raw_limit is not None:
        # for LIMIT ALL there is no second ival, but instead an "isnull" member that is set to true
        raw_limit = raw_limit["A_Const"]["ival"]
        nrows: int | None = raw_limit["ival"] if "ival" in raw_limit else None
    else:
        nrows = None
    if raw_offset is not None:
        offset: int = raw_offset["A_Const"]["ival"].get("ival", 0)
    else:
        offset = None

    return Limit(limit=nrows, offset=offset)


def _pglast_parse_setop(pglast_data: dict, *, parent_namespace: QueryNamespace) -> SetQuery:  # type: ignore # noqa: F821
    """Handler method to parse set operations.

    This method assumes that the given query is indeed a set operation and will fail otherwise.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the entire query. The method takes care of accessing the appropriate keys by itself, no preparation
        of the ``SelectStmt`` is necessary.
    parent_namespace : QueryNamespace
        The tables and columns that can be referenced by expressions in the query.

    Returns
    -------
    SetOperationClause
        The parsed set clause
    """
    if "withClause" in pglast_data:
        with_clause = _pglast_parse_ctes(pglast_data["withClause"], parent_namespace=parent_namespace)
    else:
        with_clause = None

    left_query = _pglast_parse_query(pglast_data["larg"], namespace=parent_namespace.open_nested(source="setop"))
    right_query = _pglast_parse_query(pglast_data["rarg"], namespace=parent_namespace.open_nested(source="setop"))

    match pglast_data["op"]:
        case "SETOP_UNION":
            operator = SetOperator.UnionAll if pglast_data.get("all", False) else SetOperator.Union
        case "SETOP_INTERSECT":
            operator = SetOperator.Intersect
        case "SETOP_EXCEPT":
            operator = SetOperator.Except
        case _:
            raise ParserError("Unknown set operation: " + pglast_data["op"])

    if "sortClause" in pglast_data:
        order_clause = _pglast_parse_orderby(pglast_data["sortClause"], namespace=parent_namespace)
    else:
        order_clause = None

    if pglast_data["limitOption"] == "LIMIT_OPTION_COUNT":
        limit_clause = _pglast_parse_limit(pglast_data, namespace=parent_namespace)
    else:
        limit_clause = None

    parent_namespace.determine_output_shape(None)
    return SetQuery(left_query, right_query, set_operation=operator,
                    cte_clause=with_clause, orderby_clause=order_clause, limit_clause=limit_clause)


def _pglast_parse_explain(pglast_data: dict) -> tuple[Optional[Explain], dict]:
    """Handler method to extract the *EXPLAIN* clause from a query.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the entire query. The method takes care of accessing the appropriate keys by itself, no preparation
        of the dictionary is necessary.

    Returns
    -------
    tuple[Optional[Explain], dict]
        The parsed explain clause if one exists as well as the wrapped query. The query representation should be used for all
        further parsing steps.
    """
    if "ExplainStmt" not in pglast_data:
        return None, pglast_data

    pglast_data = pglast_data["ExplainStmt"]
    explain_options: list[dict] = pglast_data.get("options", [])

    use_analyze = False
    output_format = "TEXT"
    for option in explain_options:
        definition: dict = option["DefElem"]
        match definition["defname"]:
            case "analyze":
                use_analyze = True
            case "format":
                output_format = definition["arg"]["String"]["sval"]
            case _:
                raise ParserError("Unknown explain option: " + str(definition))

    explain_clause = Explain(use_analyze, output_format)
    return explain_clause, pglast_data["query"]


def _pglast_parse_query(stmt: dict, *, namespace: QueryNamespace) -> SelectStatement:  # type: ignore # noqa: F821
    """Main entry point into the parsing logic.

    This function takes a single SQL SELECT query and provides the corresponding `SqlQuery` object.
    While parsing the different expressions, columns are automatically bound to their tables if they use qualified names.
    Otherwise, they are inferred from the database schema if one is given. If no schema is provided, the column will be
    left unbound.

    To keep track of tables that columns can bind themselves to, the `available_tables` dictionary is used. See the parameter
    documentation for more details.
    Likewise, the `resolved_columns` act as a cache for columns that have already been bound once. This is useful to prevent
    redundant lookups in the schema.

    Parameters
    ----------
    stmt : dict
        The JSON representation of the query. This should be the contents of the ``SelectStmt`` key in the JSON dictionary.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps *(table, column)* pairs to
        their respective column objects. If columns do not use a qualified name, the *table* will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

    Returns
    -------
    SelectStatement
        The parsed query
    """
    if stmt["op"] != "SETOP_NONE":
        return _pglast_parse_setop(stmt, parent_namespace=namespace)

    clauses = []

    if "withClause" in stmt:
        with_clause = _pglast_parse_ctes(stmt["withClause"], parent_namespace=namespace)
        clauses.append(with_clause)

    if "fromClause" in stmt:
        from_clause = _pglast_parse_from(stmt["fromClause"], namespace=namespace)
        clauses.append(from_clause)

    # Each query is guaranteed to have a SELECT clause, so we can just parse it straight away
    select_clause = _pglast_parse_select(stmt, namespace=namespace)
    clauses.append(select_clause)

    if "whereClause" in stmt:
        where_clause = _pglast_parse_where(stmt["whereClause"], namespace=namespace)
        clauses.append(where_clause)

    if "groupClause" in stmt:
        group_clause = _pglast_parse_groupby(stmt["groupClause"], namespace=namespace)
        clauses.append(group_clause)

    if "havingClause" in stmt:
        having_clause = _pglast_parse_having(stmt["havingClause"], namespace=namespace)
        clauses.append(having_clause)

    if "sortClause" in stmt:
        order_clause = _pglast_parse_orderby(stmt["sortClause"], namespace=namespace)
        clauses.append(order_clause)

    if stmt["limitOption"] == "LIMIT_OPTION_COUNT":
        limit_clause = _pglast_parse_limit(stmt, namespace=namespace)
        clauses.append(limit_clause)

    return build_query(clauses)


def _pglast_parse_set_commands(pglast_data: list[dict]) -> tuple[list[str], list[dict]]:
    """Handler method to parse all *SET* commands that precede the actual query.

    Parameters
    ----------
    pglast_data : list[dict]
        JSON encoding of the entire query. The method takes care of accessing the appropriate keys by itself, no preparation
        of the dictionary is necessary.

    Returns
    -------
    tuple[list[str], list[dict]]
        The parsed *SET* commands as a list of strings and the remaining query data. The query data is "forwarded" to the first
        encoding that does not represent a *SET* command.
    """
    prep_stmts: list[str] = []

    for i, item in enumerate(pglast_data):
        stmt: dict = item["stmt"]
        if "VariableSetStmt" not in stmt:
            break

        var_set_stmt: dict = stmt["VariableSetStmt"]
        if var_set_stmt["kind"] != "VAR_SET_VALUE":
            raise ParserError(f"Unknown variable set option: {var_set_stmt}")
        var_name = var_set_stmt["name"]
        var_value = var_set_stmt["args"][0]["A_Const"]["sval"]["sval"]

        parsed_stmt = f"SET {var_name} TO '{var_value}';"
        prep_stmts.append(parsed_stmt)

    return prep_stmts, pglast_data[i:]


def _parse_hint_block(raw_query: str, *, set_cmds: list[str], _current_hint_text: list[str] = None) -> Optional[Hint]:
    """Handler method to extract the hint block (i.e. preceding comments) from a query

    Parameters
    ----------
    raw_query : str
        The query text that was passed to the parser. We require access to the raw query, because the PG parser ignores all
        comments and does not represent them in the AST in any way.
    set_cmds: list[str]
        *SET* commands that have already been parsed. These will be added to the hint block.
    _current_hint_text : list[str], optional
        Internal parameter to keep track of the current hint text. This is used because the parsing logic uses a recursive
        implementation.

    Returns
    -------
    Optional[Hint]
        The hint block if any hints were found, or *None* otherwise.
    """
    _current_hint_text = _current_hint_text or []

    raw_query = raw_query.lstrip()
    block_hint = raw_query.startswith("/*")
    line_hint = raw_query.startswith("--")
    if not block_hint and not line_hint:
        prep_stms = "\n".join(set_cmds)
        hints = "\n".join(_current_hint_text)
        return Hint(prep_stms, hints) if prep_stms or hints else None

    if line_hint:

        line_end = raw_query.find("\n")
        if line_end == -1:
            # should never be raised b/c parsing should have failed already at this point
            raise ParserError(f"Unterminated line comment: {raw_query}")

        line_comment = raw_query[:line_end].strip()
        _current_hint_text.append(line_comment)
        return _parse_hint_block(raw_query[line_end:], set_cmds=set_cmds, _current_hint_text=_current_hint_text)

    # must be block hint
    block_end = raw_query.find("*/")
    if block_end == -1:
        # should never be raised b/c parsing should have failed already at this point
        raise ParserError(f"Unterminated block comment: {raw_query}")

    block_comment = raw_query[:block_end+2].strip()
    _current_hint_text.append(block_comment)
    return _parse_hint_block(raw_query[block_end+2:], set_cmds=set_cmds, _current_hint_text=_current_hint_text)


def _apply_extra_clauses(parsed: SelectStatement, *, hint: Optional[Hint],
                         explain_clause: Optional[Explain]) -> SelectStatement:
    clauses = list(parsed.clauses())
    if hint:
        clauses.append(hint)
    if explain_clause:
        clauses.append(explain_clause)
    return build_query(clauses)


@overload
def parse_query(query: str, *, include_hints: bool = True, bind_columns: bool | None = None,
                db_schema: Optional[DBCatalog] = None) -> SqlQuery:
    ...


@overload
def parse_query(query: str, *, accept_set_query: bool, include_hints: bool = True,
                bind_columns: Optional[bool] = None,
                db_schema: Optional[DBCatalog] = None) -> SelectStatement:
    ...


def parse_query(query: str, *, accept_set_query: bool = False, include_hints: bool = True,
                bind_columns: Optional[bool] = None,
                db_schema: Optional[DBCatalog] = None) -> SelectStatement:
    """Parses a query string into a proper `SqlQuery` object.

    During parsing, the appropriate type of SQL query (i.e. with implicit, explicit or mixed *FROM* clause) will be
    inferred automatically. Therefore, this method can potentially return a subclass of `SqlQuery`.

    Once the query has been transformed, a text-based binding process is executed. During this process, the referenced
    tables are normalized such that column references using the table alias are linked to the correct tables that are
    specified in the *FROM* clause (see the module-level documentation for an example). The parsing process can
    optionally also involve a binding process based on the schema of a live database. This is important for all
    remaining columns where the text-based parsing was not possible, e.g. because the column was specified without a
    table alias.

    Parameters
    ----------
    query : str
        The query to parse
    accept_set_query : bool, optional
        Whether set queries are a valid result of the parsing process. If this is *False* (the default), an error will be
        raised if the input query is a set query. This implies that the result of the parsing process is always a `SqlQuery`
        instance. Otherwise, the result can also be a `SetQuery` instance.
    include_hints : bool, optional
        Whether to include hints in the parsed query. If this is *True* (the default), any preceding comments in the query
        text will be parsed as a hint block. Otherwise, these comments are simply ignored.
    bind_columns : bool | None, optional
        Whether to use *live binding*. This does not control the text-based binding, which is always performed. If this
        parameter is *None* (the default), the global `auto_bind_columns` variable will be queried. Depending on its
        value, live binding will be performed or not.
    db_schema : Optional[DBCatalog], optional
        For live binding, this indicates the database to use. If this is *None* (the default), the database will be
        tried to extract from the `DatabasePool`

    Returns
    -------
    SqlQuery
        The parsed SQL query.
    """
    # NOTE: this documentation is a 1:1 copy of qal.parse_query. Both should be kept in sync.
    if db_schema is None and (bind_columns or (bind_columns is None and auto_bind_columns)):
        from ..db import DatabasePool  # local import to prevent circular imports
        db_schema = None if DatabasePool.get_instance().empty() else DatabasePool.get_instance().current_database().schema()

    pglast_data = json.loads(pglast.parser.parse_sql_json(query))
    stmts = pglast_data["stmts"]

    set_cmds, stmts = _pglast_parse_set_commands(stmts)
    if len(stmts) != 1:
        raise ValueError("Parser can only support single-statement queries for now")
    raw_query: dict = stmts[0]["stmt"]

    if "ExplainStmt" in raw_query:
        explain_clause, raw_query = _pglast_parse_explain(raw_query)
    else:
        explain_clause = None

    if "SelectStmt" not in raw_query:
        raise ValueError("Cannot parse non-SELECT queries")
    stmt = raw_query["SelectStmt"]

    parsed_query = _pglast_parse_query(stmt, namespace=QueryNamespace.empty(db_schema))
    if not accept_set_query and isinstance(query, SetQuery):
        raise ParserError("Input query is a set query")

    hint = _parse_hint_block(query, set_cmds=set_cmds) if include_hints else None
    parsed_query = _apply_extra_clauses(parsed_query, hint=hint, explain_clause=explain_clause)

    return parsed_query


class ParserError(RuntimeError):
    """An error that is raised when parsing fails."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


def load_table_json(json_data: dict | str) -> Optional[TableReference]:
    """Re-creates a table reference from its JSON encoding.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*)

    Returns
    -------
    Optional[TableReference]
        The actual table. If the dictionary is empty or otherwise invalid, *None* is returned.
    """
    if not json_data:
        return None
    json_data = json_data if isinstance(json_data, dict) else json.loads(json_data)
    return TableReference(json_data.get("full_name", ""), json_data.get("alias", ""),
                          virtual=json_data.get("virtual", False), schema=json_data.get("schemaname", None))


def load_column_json(json_data: dict | str) -> Optional[ColumnReference]:
    """Re-creates a column reference from its JSON encoding.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*)

    Returns
    -------
    Optional[ColumnReference]
        The actual column. It the dictionary is empty or otherwise invalid, *None* is returned.
    """
    if not json_data:
        return None
    json_data = json_data if isinstance(json_data, dict) else json.loads(json_data)
    return ColumnReference(json_data.get("column"), load_table_json(json_data.get("table", None)))


def load_expression_json(json_data: dict | str) -> Optional[SqlExpression]:
    """Re-creates an arbitrary SQL expression from its JSON encoding.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*)

    Returns
    -------
    Optional[SqlExpression]
        The actual expression. If the dictionary is empty or *None*, *None* is returned. Notice that in case of
        malformed data, errors are raised.
    """
    if not json_data:
        return None
    json_data = json_data if isinstance(json_data, dict) else json.loads(json_data)

    tables = [load_table_json(table_data) for table_data in json_data.get("tables", [])]
    expression_str = json_data["expression"]
    if not tables:
        emulated_query = f"SELECT {expression_str}"
    else:
        from_clause_str = ", ".join(str(tab) for tab in tables)
        emulated_query = f"SELECT {expression_str} FROM {from_clause_str}"

    parsed_query = parse_query(emulated_query)
    return parsed_query.select_clause.targets[0].expression


def load_predicate_json(json_data: dict | str) -> Optional[AbstractPredicate]:
    """Re-creates an arbitrary predicate from its JSON encoding.

    Parameters
    ----------
    json_data : dict | str
        Either the JSON dictionary, or a string encoding of the dictionary (which will be parsed by *json.loads*)

    Returns
    -------
    Optional[AbstractPredicate]
        The actual predicate. If the dictionary is empty or *None*, *None* is returned. Notice that in case of
        malformed data, errors are raised.

    Raises
    ------
    KeyError
        If the encoding does not specify the tables that are referenced in the predicate
    KeyError
        If the encoding does not contain the actual predicate
    """
    if not json_data:
        return None
    json_data = json_data if isinstance(json_data, dict) else json.loads(json_data)

    tables = [load_table_json(table_data) for table_data in json_data.get("tables", [])]
    if not tables:
        raise KeyError("Predicate needs at least one table!")
    from_clause_str = ", ".join(str(tab) for tab in tables)
    predicate_str = json_data["predicate"]
    emulated_query = f"SELECT * FROM {from_clause_str} WHERE {predicate_str}"
    parsed_query = parse_query(emulated_query)
    return parsed_query.where_clause.predicate
