"""The parser constructs `SqlQuery` objects from query strings.

Other than the parsing itself, the process will also execute a basic column binding process. For example, consider
a query like ``SELECT * FROM R WHERE R.a = 42``. In this case, the binding only affects the column reference ``R.a``
and sets the table of that column to ``R``. This binding based on column and table names is always performed.

If the table cannot be inferred based on the column name (e.g. for a query like ``SELECT * FROM R, S WHERE a = 42``), a
second binding phase can be executed. This binding needs a working database connection and queries the database schema
to detect the correct tables for each column. Whether the second phase should also be executed by default can be
configured system-wide by setting the `auto_bind_columns` variable.

Notes
-----
The parsing itself is based on the mo-sql-parsing project that implements a SQL -> JSON/dict conversion (which at
times is more akin to a tokenization than an actual parsing). Our parser implementation than takes such a JSON
representation and generates the more verbose structures of the qal. There exists a Jupyter notebook called
*MoSQLParsingTests* in the *analysis* directory that shows the output emitted by mo-sql-parsing for different SQL query
features.

References
----------

.. mo-sql-parsing project: https://github.com/klahnakoski/mo-sql-parsing Thanks a lot for maintaining this fantastic
   tool and the great support!
"""
from __future__ import annotations

import copy
import json
import re
import warnings
from typing import Any, Optional

import mo_sql_parsing as mosp
import pglast

from . import transform
from ._core import (
    TableReference, ColumnReference,
    SelectType, JoinType,
    CompoundOperators, MathematicalSqlOperators, LogicalSqlOperators, SqlOperator,
    BaseProjection, WithQuery, DirectTableSource, JoinTableSource, SubqueryTableSource, OrderByExpression,
    SqlQuery, ImplicitSqlQuery, ExplicitSqlQuery, MixedSqlQuery,
    Select, Where, From, GroupBy, Having, OrderBy, Limit, CommonTableExpression, ImplicitFromClause, ExplicitFromClause,
    UnionClause, IntersectClause, ExceptClause,
    AbstractPredicate, BinaryPredicate, InPredicate, BetweenPredicate, CompoundPredicate, UnaryPredicate,
    SqlExpression, StarExpression, StaticValueExpression, ColumnExpression, SubqueryExpression, CastExpression,
    MathematicalExpression, FunctionExpression, BooleanExpression, WindowExpression, CaseExpression,
    build_query
)
from .transform import QueryType
from .. import util

auto_bind_columns: bool = False
"""Indicates whether the parser should use the database catalog to obtain column bindings."""


def _pglast_parse_colref(pglast_data: dict, *, available_tables: dict[str, TableReference],
                         resolved_columns: dict[str, ColumnReference],
                         schema: Optional["DatabaseSchema"]) -> ColumnReference:  # type: ignore # noqa: F821
    fields = pglast_data["fields"]
    if len(fields) > 2:
        raise ParserError("Unknown column reference format: " + str(pglast_data))

    if len(fields) == 2:
        tab, col = fields
        tab: str = tab["String"]["sval"]
        col: str = col["String"]["sval"]
        parsed_table = available_tables[tab]
        parsed_column = ColumnReference(col, parsed_table)
        resolved_columns[col] = parsed_column
        return parsed_column

    # at this point, we must have a single unbound column parameter
    col: str = fields[0]["String"]["sval"]
    if col in resolved_columns:
        return resolved_columns[col]
    if not schema:
        return ColumnReference(col, None)

    try:
        resolved_table = schema.lookup_column(col, available_tables.values())
    except ValueError:
        raise ParserError("Could not resolve column reference: " + col)

    parsed_column = ColumnReference(col, resolved_table)
    resolved_columns[col] = parsed_column
    return parsed_column


def _pglast_parse_const(pglast_data: dict) -> StaticValueExpression:
    pglast_data.pop("location", None)
    valtype = util.dicts.key(pglast_data)
    match valtype:
        case "isnull":
            return StaticValueExpression.null()
        case "ival":
            val = pglast_data["ival"]["ival"]
            return StaticValueExpression(val)
        case "fval":
            val = pglast_data["fval"]["fval"]
            return StaticValueExpression(float(val))
        case "sval":
            return StaticValueExpression(pglast_data["sval"]["sval"])
        case "boolval":
            val = False if not pglast_data["boolval"]["boolval"] else True
            return StaticValueExpression(val)
        case _:
            raise ParserError("Unknown constant type: " + str(pglast_data))


def _pglast_parse_expression(pglast_data: dict, *, available_tables: dict[str, TableReference],
                             resolved_columns: dict[str, ColumnReference],
                             schema: Optional["DBSchema"]) -> SqlExpression:  # type: ignore # noqa: F821
    pglast_data.pop("location", None)
    expression_key = util.dicts.key(pglast_data)

    match expression_key:

        case "ColumnRef":
            column = _pglast_parse_colref(pglast_data["ColumnRef"], available_tables=available_tables,
                                          resolved_columns=resolved_columns, schema=schema)
            return ColumnExpression(column)

        case "A_Const":
            return _pglast_parse_const(pglast_data["A_Const"])

        case _:
            raise ParserError("Unknown expression type: " + str(pglast_data))


def _pglast_try_select_star(target: dict) -> Optional[Select]:
    if "ColumnRef" not in target:
        return None
    fields = target["ColumnRef"]["fields"]
    if len(fields) != 1:
        # multiple fields are used for qualified column references. This is definitely not a SELECT * query, so exit
        return None
    colref = fields[0]
    return Select.star() if "A_Star" in colref else None


def _pglast_parse_select(targetlist: list, *, distinct: bool,
                         available_tables: dict[str, TableReference],
                         resolved_columns: dict[str, ColumnReference],
                         schema: Optional["DBSchema"]) -> Select:  # type: ignore # noqa: F821
    # first, try for SELECT * queries
    if len(targetlist) == 1:
        target = targetlist[0]["ResTarget"]["val"]
        select_star = _pglast_try_select_star(target)

        if select_star:
            return select_star
        # if this is not a SELECT * query, we can continue with the regular parsing

    targets: list[BaseProjection] = []
    for target in targetlist:
        expression = _pglast_parse_expression(target["ResTarget"]["val"],
                                              available_tables=available_tables,
                                              resolved_columns=resolved_columns, schema=schema)
        alias = target["ResTarget"].get("name", "")
        projection = BaseProjection(expression, alias)
        targets.append(projection)

    return Select(targets, projection_type=SelectType.SelectDistinct if distinct else SelectType.Select)


def _pglast_parse_rangevar(rangevar: dict) -> TableReference:
    name = rangevar["relname"]
    alias = rangevar["alias"]["aliasname"] if "alias" in rangevar else None
    return TableReference(name, alias)


def _pglast_parse_from(from_clause: list, *,
                       available_tables: dict[str, TableReference],
                       resolved_columns: dict[str, ColumnReference],
                       schema: Optional["DBSchema"]) -> From:  # type: ignore # noqa: F821
    contains_join = False
    contains_mixed = False
    contains_subquery = False

    table_sources = []
    for entry in from_clause:
        entry.pop("location", None)
        entry_type = util.dicts.key(entry)

        match entry_type:

            case "RangeVar":
                if contains_join:
                    contains_mixed = True
                table = _pglast_parse_rangevar(entry["RangeVar"])
                available_tables[table.identifier()] = table
                table_sources.append(DirectTableSource(table))

            case _:
                raise ParserError("Unknow FROM clause entry: " + str(entry))

    if not contains_join and not contains_mixed and not contains_subquery:
        return ImplicitFromClause(table_sources)
    if contains_join and not contains_mixed and not contains_subquery:
        return ExplicitFromClause(table_sources)
    return From(table_sources)


PglastOperatorMap: dict[str, SqlOperator] = {
    "=": LogicalSqlOperators.Equal,
    "<": LogicalSqlOperators.Less,
    "<=": LogicalSqlOperators.LessEqual,
    ">": LogicalSqlOperators.Greater,
    ">=": LogicalSqlOperators.GreaterEqual,
    "<>": LogicalSqlOperators.NotEqual,
    "!=": LogicalSqlOperators.NotEqual,

    "AND_EXPR": CompoundOperators.And,
    "OR_EXPR": CompoundOperators.Or,
    "NOT_EXPR": CompoundOperators.Not,

    "+": MathematicalSqlOperators.Add,
    "-": MathematicalSqlOperators.Subtract,
    "*": MathematicalSqlOperators.Multiply,
    "/": MathematicalSqlOperators.Divide,
    "%": MathematicalSqlOperators.Modulo
}


def _pglast_parse_operator(pglast_data: list) -> LogicalSqlOperators:
    if len(pglast_data) != 1:
        raise ParserError("Unknown operator format: " + str(pglast_data))
    operator = pglast_data[0]
    if "String" not in operator or "sval" not in operator["String"]:
        raise ParserError("Unknown operator format: " + str(pglast_data))
    sval = operator["String"]["sval"]
    if sval not in PglastOperatorMap:
        raise ParserError("Operator not yet in target map: " + sval)
    return PglastOperatorMap[sval]


def _pglast_parse_predicate(pglast_data: dict, available_tables: dict[str, TableReference],
                            resolved_columns: dict[str, ColumnReference],
                            schema: Optional["DBSchema"]) -> AbstractPredicate:  # type: ignore # noqa: F821
    pglast_data.pop("location", None)
    expr_key = util.dicts.key(pglast_data)
    match expr_key:

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_OP":
            expression = pglast_data["A_Expr"]
            operator = _pglast_parse_operator(expression["name"])
            left = _pglast_parse_expression(expression["lexpr"], available_tables=available_tables,
                                            resolved_columns=resolved_columns, schema=schema)
            right = _pglast_parse_expression(expression["rexpr"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AXPR_LIKE":
            expression = pglast_data["A_Expr"]
            operator = (LogicalSqlOperators.Like if expression["name"][0]["String"]["sval"] == "~~"
                        else LogicalSqlOperators.NotLike)
            left = _pglast_parse_expression(expression["lexpr"], available_tables=available_tables,
                                            resolved_columns=resolved_columns, schema=schema)
            right = _pglast_parse_expression(expression["rexpr"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_ILIKE":
            expression = pglast_data["A_Expr"]
            operator = (LogicalSqlOperators.ILike if expression["name"][0]["String"]["sval"] == "~~*"
                        else LogicalSqlOperators.NotILike)
            left = _pglast_parse_expression(expression["lexpr"], available_tables=available_tables,
                                            resolved_columns=resolved_columns, schema=schema)
            right = _pglast_parse_expression(expression["rexpr"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_BETWEEN":
            expression = pglast_data["A_Expr"]
            left = _pglast_parse_expression(expression["lexpr"], available_tables=available_tables,
                                            resolved_columns=resolved_columns, schema=schema)
            raw_interval = expression["rexpr"]["List"]["items"]
            if len(raw_interval) != 2:
                raise ParserError("Invalid BETWEEN interval: " + str(raw_interval))
            lower = _pglast_parse_expression(raw_interval[0], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
            upper = _pglast_parse_expression(raw_interval[1], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
            return BetweenPredicate(left, (lower, upper))

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_IN":
            expression = pglast_data["A_Expr"]
            left = _pglast_parse_expression(expression["lexpr"], available_tables=available_tables,
                                            resolved_columns=resolved_columns, schema=schema)
            raw_values = expression["rexpr"]["List"]["items"]
            values = [_pglast_parse_expression(value, available_tables=available_tables,
                                               resolved_columns=resolved_columns, schema=schema)
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
            operator = PglastOperatorMap[expression["boolop"]]
            children = [_pglast_parse_predicate(child,
                                                available_tables=available_tables,
                                                resolved_columns=resolved_columns,
                                                schema=schema)
                        for child in expression["args"]]
            return CompoundPredicate(operator, children)

        case "NullTest":
            expression = pglast_data["NullTest"]
            testexpr = _pglast_parse_expression(expression["arg"],
                                                available_tables=available_tables,
                                                resolved_columns=resolved_columns,
                                                schema=schema)
            operation = LogicalSqlOperators.Is if expression["nulltesttype"] == "IS_NULL" else LogicalSqlOperators.IsNot
            return BinaryPredicate(operation, testexpr, StaticValueExpression.null())

        case "SubLink":
            expression = pglast_data["SubLink"]
            sublink_type = expression["subLinkType"]

            subquery = _pglast_parse_query(expression["subselect"]["SelectStmt"],
                                           available_tables=available_tables,
                                           resolved_columns=resolved_columns,
                                           schema=schema)
            if sublink_type == "EXISTS_SUBLINK":
                return UnaryPredicate.exists(subquery)

            testexpr = _pglast_parse_expression(expression["testexpr"],
                                                available_tables=available_tables,
                                                resolved_columns=resolved_columns,
                                                schema=schema)

            if sublink_type == "ANY_SUBLINK" and "operName" not in expression:
                return InPredicate.subquery(testexpr, subquery)

            if sublink_type == "ANY_SUBLINK":
                operator = PglastOperatorMap[expression["operName"]]
                subquery_expression = FunctionExpression.any_func(subquery)
                return BinaryPredicate(operator, testexpr, subquery_expression)
            elif sublink_type == "ALL_SUBLINK":
                operator = PglastOperatorMap[expression["operName"]]
                subquery_expression = FunctionExpression.all_func(subquery)
                return BinaryPredicate(operator, testexpr, subquery_expression)
            else:
                raise NotImplementedError("Subquery handling is not yet implemented")

        case _:
            raise ParserError("Unknown predicate type: " + str(pglast_data))


def _pglast_parse_where(where_clause: dict, *,
                        available_tables: dict[str, TableReference],
                        resolved_columns: dict[str, ColumnReference],
                        schema: Optional["DBSchema"]) -> Optional[Where]:  # type: ignore # noqa: F821
    predicate = _pglast_parse_predicate(where_clause, available_tables=available_tables,
                                        resolved_columns=resolved_columns, schema=schema)
    return Where(predicate)


def _pglast_parse_query(stmt: dict, *, available_tables: dict[str, TableReference],
                        resolved_columns: dict[str, ColumnReference],
                        schema: Optional["DBSchema"]) -> SqlQuery:  # type: ignore # noqa: F821
    clauses = []

    if "fromClause" in stmt:
        from_clause = _pglast_parse_from(stmt["fromClause"], available_tables=available_tables,
                                         resolved_columns=resolved_columns, schema=schema)
        clauses.append(from_clause)

    # Each query is guaranteed to have a SELECT clause, so we can just parse it straight away
    select_distinct = "distinctClause" in stmt
    select_clause = _pglast_parse_select(stmt["targetList"], distinct=select_distinct,
                                         available_tables=available_tables, resolved_columns=resolved_columns, schema=schema)
    clauses.append(select_clause)

    if "whereClause" in stmt:
        where_clause = _pglast_parse_where(stmt["whereClause"], available_tables=available_tables,
                                           resolved_columns=resolved_columns, schema=schema)
        clauses.append(where_clause)

    return build_query(clauses)


def _pglast_based_query_parser(query: str, *, bind_columns: bool | None = None,
                               db_schema: Optional["DatabaseSchema"] = None) -> SqlQuery:  # type: ignore # noqa: F821
    warnings.warn("pglast-based query parsing is still experimental and might contain bugs.", FutureWarning)

    if db_schema is None and (bind_columns or (bind_columns is None and auto_bind_columns)):
        from ..db import DatabasePool  # local import to prevent circular imports
        db_schema = None if DatabasePool.get_instance().empty() else DatabasePool.get_instance().current_database().schema()

    pglast_data = json.loads(pglast.parser.parse_sql_json(query))
    stmts = pglast_data["stmts"]
    if len(stmts) != 1:
        raise ValueError("Parser can only support single-statement queries for now")
    raw_query = stmts[0]["stmt"]
    if "SelectStmt" not in raw_query:
        raise ValueError("Cannot parse non-SELECT queries")
    stmt = raw_query["SelectStmt"]

    parsed_query = _pglast_parse_query(stmt, available_tables={}, resolved_columns={}, schema=db_schema)
    return parsed_query


def parse_query(query: str, *, bind_columns: bool | None = None,
                db_schema: Optional["DatabaseSchema"] = None,  # type: ignore # noqa: F821
                _skip_all_binding: bool = False) -> SqlQuery:
    """Parses a query string into a proper `SqlQuery` object.

    During parsing, the appropriate type of SQL query (i.e. with implicit, explicit or mixed ``FROM`` clause) will be
    inferred automatically. Therefore, this method can potentially return a subclass of `SqlQuery`.

    Once the query has been transformed, a text-based binding process is executed. During this process, the referenced
    tables are normalized such that column references using the table alias are linked to the correct tables that are
    specified in the ``FROM`` clause (see the module-level documentation for an example). The parsing process can
    optionally also involve a binding process based on the schema of a live database. This is important for all
    remaining columns where the text-based parsing was not possible, e.g. because the column was specified without a
    table alias.

    Parameters
    ----------
    query : str
        The query to parse
    bind_columns : bool | None, optional
        Whether to use *live binding*. This does not control the text-based binding, which is always performed. If this
        parameter is ``None`` (the default), the global `auto_bind_columns` variable will be queried. Depending on its
        value, live binding will be performed or not.
    db_schema : Optional[DatabaseSchema], optional
        For live binding, this indicates the database to use. If this is ``None`` (the default), the database will be
        tried to extract from the `DatabasePool`

    Returns
    -------
    SqlQuery
        The parsed SQL query.
    """
    # NOTE: this documentation is a 1:1 copy of qal.parse_query. Both should be kept in sync.
    return _pglast_based_query_parser(query, bind_columns=bind_columns, db_schema=db_schema, _skip_all_binding=_skip_all_binding)


class ParserError(RuntimeError):
    """An error that is raised when parsing fails."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


def load_table_json(json_data: dict) -> Optional[TableReference]:
    """Re-creates a table reference from its JSON encoding.

    Parameters
    ----------
    json_data : dict
        The encoded table

    Returns
    -------
    Optional[TableReference]
        The actual table. If the dictionary is empty or otherwise invalid, *None* is returned.
    """
    if not json_data:
        return None
    return TableReference(json_data.get("full_name", ""), json_data.get("alias", ""))


def load_column_json(json_data: dict) -> Optional[ColumnReference]:
    """Re-creates a column reference from its JSON encoding.

    Parameters
    ----------
    json_data : dict
        The encoded column

    Returns
    -------
    Optional[ColumnReference]
        The actual column. It the dictionary is empty or otherwise invalid, *None* is returned.
    """
    if not json_data:
        return None
    return ColumnReference(json_data.get("column"), load_table_json(json_data.get("table", None)))


def load_predicate_json(json_data: dict) -> Optional[AbstractPredicate]:
    """Re-creates an arbitrary predicate from its JSON encoding.

    Parameters
    ----------
    json_data : dict
        The encoded predicate

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
    tables = [load_table_json(table_data) for table_data in json_data.get("tables", [])]
    if not tables:
        raise KeyError("Predicate needs at least one table!")
    from_clause_str = ", ".join(str(tab) for tab in tables)
    predicate_str = json_data["predicate"]
    emulated_query = f"SELECT * FROM {from_clause_str} WHERE {predicate_str}"
    parsed_query = parse_query(emulated_query)
    return parsed_query.where_clause.predicate
