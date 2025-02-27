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

import json
from typing import Optional, overload

import pglast

from ._qal import (
    ColumnReference,
    SelectType, JoinType,
    CompoundOperator, MathOperator, LogicalOperator, SqlOperator, SetOperator,
    BaseProjection, OrderByExpression,
    WithQuery, TableSource, DirectTableSource, JoinTableSource, SubqueryTableSource,
    ValuesList, ValuesTableSource, ValuesWithQuery,
    SqlQuery, SetQuery, SelectStatement, ImplicitSqlQuery, ExplicitSqlQuery, MixedSqlQuery,
    Select, Where, From, GroupBy, Having, OrderBy, Limit, CommonTableExpression, ImplicitFromClause, ExplicitFromClause,
    Hint, Explain,
    AbstractPredicate, BinaryPredicate, InPredicate, BetweenPredicate, CompoundPredicate, UnaryPredicate,
    SqlExpression, StarExpression, StaticValueExpression, ColumnExpression, SubqueryExpression, CastExpression,
    MathematicalExpression, FunctionExpression, WindowExpression, CaseExpression, ArrayAccessExpression,
    build_query
)
from .._core import TableReference, normalize
from .. import util

auto_bind_columns: bool = False
"""Indicates whether the parser should use the database catalog to obtain column bindings."""


def _pglast_is_actual_colref(pglast_data: dict) -> bool:
    """Checks, whether a apparent column reference is actually a column reference and not a star expression in disguise.

    pglast represents both column references such as *R.a* or *a* as well as star expressions like *R.** as ``ColumnRef``
    dictionaries, hence we need to make sure we are actually parsing the right thing. This method takes care of distinguishing
    the two cases.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the potential column

    Returns
    -------
    bool
        **True** if this is an actual column reference, **False** if this is a star expression.
    """
    fields: list[dict] = pglast_data["fields"]
    if len(fields) == 1:
        return "A_Star" not in fields[0]
    if len(fields) == 2:
        would_be_col: str = fields[1]
        return "A_Star" not in would_be_col

    would_be_col: str = fields[0]["String"]["sval"]
    return not would_be_col.endswith("*")


def _pglast_create_bound_colref(tab: str, col: str, available_tables: dict[str, TableReference],
                                resolved_columns: dict[tuple[str, str], ColumnReference]) -> ColumnReference:
    """Creates a new reference to a column with known binding info.

    Parameters
    ----------
    tab : str
        The table to which to bind
    col : str
        The column to bind
    available_tables : dict[str, TableReference]
        Candidates for the binding table
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Already resolved columns

    Returns
    -------
    ColumnReference
        The new column reference
    """
    parsed_table = available_tables.get(normalize(tab), None)
    if not parsed_table:
        raise ParserError("Table not found: " + tab)
    parsed_column = ColumnReference(col, parsed_table)
    resolved_columns[(tab, col)] = parsed_column
    return parsed_column


def _pglast_parse_colref(pglast_data: dict, *, available_tables: dict[str, TableReference],
                         resolved_columns: dict[tuple[str, str], ColumnReference],
                         schema: Optional["DatabaseSchema"]) -> ColumnReference:  # type: ignore # noqa: F821
    """Handler method to parse column references in the query.

    The column will be bound to its table if possible. This binding process uses the following rules:

    - if the columns has already been resolved as part of an earlier parsing step, this column is re-used from the
      `resolved_columns`
    - if the column is specified in qualified syntax (i.e. **table.column**), the table is directly inferred from the
      `available_tables`.
    - if the column is not qualified, but a `schema` is given, this schema is used together with the candidates from
      `available_tables` to lookup the appropriate table.
    - otherwise, the column is left unbound.

    In case the schema-based binding is used, the new column is also stored in the `resolved_columns` cache.


    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the column
    available_tables : dict[str, TableReference]
        The candidate tables for all binding purposes. Maps table identifiers (i.e. full names and aliases) to the actual
        table references.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound before. This cache maps **(table, column)** pairs to their respective column
        representations. If columns do not use a qualified name, the **table** will be an empty string.
    schema : Optional[DatabaseSchema]
        The database schema info to use for live binding If this is omitted, no binding is performed and unqualified columns
        will be left without a table reference :(

    Returns
    -------
    ColumnReference
        The parsed column reference.
    """
    fields = pglast_data["fields"]
    if len(fields) > 2:
        raise ParserError("Unknown column reference format: " + str(pglast_data))

    if len(fields) == 2:
        tab, col = fields
        tab: str = tab["String"]["sval"]
        col: str = col["String"]["sval"]
        return _pglast_create_bound_colref(tab, col, available_tables, resolved_columns)

    # at this point, we must have a single column parameter. It could be unbounded, or - if quoted - bounded
    col: str = fields[0]["String"]["sval"]

    # first, check for quoted and qualified identifiers, such as "Sales.Price"
    if False and "." in col:
        tab, col = col.split(".")

        # we need to manually normalize here, because Postgres does not normalize quoted identifiers by default
        # however, this does not apply to the table, because this was already normalized as part of the CTE/FROM clause parsing
        col = normalize(col)

        return _pglast_create_bound_colref(tab, col, available_tables, resolved_columns)

    # now, we know for certain that the identifier is unqualified
    if ("", col) in resolved_columns:
        return resolved_columns[col]
    if not schema:
        return ColumnReference(col, None)

    try:
        resolved_table = schema.lookup_column(col, available_tables.values())
    except ValueError:
        raise ParserError("Could not resolve column reference: " + col)

    parsed_column = ColumnReference(col, resolved_table)
    resolved_columns[("", col)] = parsed_column
    return parsed_column


def _pglast_parse_star(pglast_data: dict, *,
                       available_tables: dict[str, TableReference]) -> StarExpression:  # type: ignore # noqa: F821
    """Handler method to parse star expressions that are potentially bounded to a specific table, e.g. *R.\\**.

    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the star expression
    available_tables : dict[str, TableReference]
        The candidate tables for all binding purposes. Maps table identifiers (i.e. full names and aliases) to the actual
        table references.

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
        return StarExpression(from_table=available_tables.get(normalize(tab), None))

    star_reference: str = fields[0]["String"]["sval"]
    if not star_reference.endswith("*") or "." not in star_reference:
        raise ParserError("Unknown star reference format: " + str(pglast_data))

    table = star_reference.split(".")[0]
    parsed_table = available_tables.get(normalize(table), None)
    return StarExpression(from_table=parsed_table)


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
    "float8": "double",

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


def _pglast_parse_case(pglast_data: dict, *, available_tables: dict[str, TableReference],
                       resolved_columns: dict[tuple[str, str], ColumnReference],
                       schema: Optional["DatabaseSchema"]) -> CaseExpression:  # type: ignore # noqa: F821
    """Handler method to parse **CASE** expressions in a query.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the **CASE** expression data. This data is extracted from the pglast data structure.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain

    Returns
    -------
    CaseExpression
        The parsed **CASE** expression.
    """
    cases: list[tuple[AbstractPredicate, SqlExpression]] = []
    for arg in pglast_data["args"]:
        current_case = _pglast_parse_predicate(arg["CaseWhen"]["expr"], available_tables=available_tables,
                                               resolved_columns=resolved_columns, schema=schema)
        current_result = _pglast_parse_expression(arg["CaseWhen"]["result"], available_tables=available_tables,
                                                  resolved_columns=resolved_columns, schema=schema)
        cases.append((current_case, current_result))

    if "defresult" in pglast_data:
        default_result = _pglast_parse_expression(pglast_data["defresult"], available_tables=available_tables,
                                                  resolved_columns=resolved_columns, schema=schema)
    else:
        default_result = None

    return CaseExpression(cases, else_expr=default_result)


def _pglast_parse_expression(pglast_data: dict, *, available_tables: dict[str, TableReference],
                             resolved_columns: dict[tuple[str, str], ColumnReference],
                             schema: Optional["DatabaseSchema"]) -> SqlExpression:  # type: ignore # noqa: F821
    """Handler method to parse arbitrary expressions in the query.

    For some more complex expressions, this method will delegate to tailored parsing methods.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the expression data. This data is extracted from the pglast data structure.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

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
            column = _pglast_parse_colref(pglast_data["ColumnRef"], available_tables=available_tables,
                                          resolved_columns=resolved_columns, schema=schema)
            return ColumnExpression(column)

        case "ColumnRef" if not _pglast_is_actual_colref(pglast_data["ColumnRef"]):
            return _pglast_parse_star(pglast_data["ColumnRef"], available_tables=available_tables)

        case "A_Const":
            return _pglast_parse_const(pglast_data["A_Const"])

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_OP":
            expression = pglast_data["A_Expr"]
            operation = _pglast_parse_operator(expression["name"])
            right = _pglast_parse_expression(expression["rexpr"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)

            if "lexpr" not in expression and operation in MathOperator:
                return MathematicalExpression(operation, right)
            elif "lexpr" not in expression:
                raise ParserError("Unknown operator format: " + str(expression))

            left = _pglast_parse_expression(expression["lexpr"], available_tables=available_tables,
                                            resolved_columns=resolved_columns, schema=schema)

            if operation in LogicalOperator:
                return BinaryPredicate(operation, left, right)

            return MathematicalExpression(operation, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] in {"AEXPR_LIKE", "AEXPR_ILIKE", "AEXPR_BETWEEN", "AEXPR_IN"}:
            # we need to parse a predicate in disguise
            predicate = _pglast_parse_predicate(pglast_data, available_tables=available_tables,
                                                resolved_columns=resolved_columns, schema=schema)
            return predicate

        case "NullTest":
            predicate = _pglast_parse_predicate(pglast_data, available_tables=available_tables,
                                                resolved_columns=resolved_columns, schema=schema)
            return predicate

        case "BoolExpr":
            predicate = _pglast_parse_predicate(pglast_data, available_tables=available_tables,
                                                resolved_columns=resolved_columns, schema=schema)
            return predicate

        case "FuncCall" if "over" not in pglast_data["FuncCall"]:  # normal functions, aggregates and UDFs
            expression: dict = pglast_data["FuncCall"]
            funcname = expression["funcname"][0]["String"]["sval"]
            distinct = expression.get("agg_distinct", False)
            if expression.get("agg_filter", False):
                filter_expr = _pglast_parse_predicate(expression["agg_filter"], available_tables=available_tables,
                                                      resolved_columns=resolved_columns, schema=schema)
            else:
                filter_expr = None

            if expression.get("agg_star", False):
                return FunctionExpression(funcname, [StarExpression()], distinct=distinct, filter_where=filter_expr)

            args = [_pglast_parse_expression(arg, available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
                    for arg in expression.get("args", [])]
            return FunctionExpression(funcname, args, distinct=distinct, filter_where=filter_expr)

        case "FuncCall" if "over" in pglast_data["FuncCall"]:  # window functions
            expression: dict = pglast_data["FuncCall"]
            funcname = expression["funcname"][0]["String"]["sval"]

            args = [_pglast_parse_expression(arg, available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
                    for arg in expression.get("args", [])]
            fn = FunctionExpression(funcname, args)

            window_spec: dict = expression["over"]

            if "partitionClause" in window_spec:
                partition = [_pglast_parse_expression(partition, available_tables=available_tables,
                                                      resolved_columns=resolved_columns, schema=schema)
                             for partition in window_spec["partitionClause"]]
            else:
                partition = None

            if "orderClause" in window_spec:
                order = _pglast_parse_orderby(window_spec["orderClause"], available_tables=available_tables,
                                              resolved_columns=resolved_columns, schema=schema)
            else:
                order = None

            if "agg_filter" in expression:
                filter_expr = _pglast_parse_expression(expression["agg_filter"], available_tables=available_tables,
                                                       resolved_columns=resolved_columns, schema=schema)
            else:
                filter_expr = None

            return WindowExpression(fn, partitioning=partition, ordering=order, filter_condition=filter_expr)

        case "CoalesceExpr":
            expression = pglast_data["CoalesceExpr"]
            args = [_pglast_parse_expression(arg, available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
                    for arg in expression["args"]]
            return FunctionExpression("coalesce", args)

        case "TypeCast":
            expression: dict = pglast_data["TypeCast"]
            casted_expression = _pglast_parse_expression(expression["arg"], available_tables=available_tables,
                                                         resolved_columns=resolved_columns, schema=schema)
            target_type = _pglast_parse_type(expression["typeName"])
            type_params = [_pglast_parse_expression(param, available_tables=available_tables,
                                                    resolved_columns=resolved_columns, schema=schema)
                           for param in expression["typeName"].get("typmods", [])]

            return CastExpression(casted_expression, target_type, type_params=type_params)

        case "CaseExpr":
            return _pglast_parse_case(pglast_data["CaseExpr"], available_tables=available_tables,
                                      resolved_columns=resolved_columns, schema=schema)

        case "SubLink" if pglast_data["SubLink"]["subLinkType"] == "EXPR_SUBLINK":
            subquery = _pglast_parse_query(pglast_data["SubLink"]["subselect"]["SelectStmt"],
                                           available_tables=dict(available_tables),
                                           resolved_columns=dict(resolved_columns),
                                           schema=schema)
            return SubqueryExpression(subquery)

        case "A_Indirection":
            expression: dict = pglast_data["A_Indirection"]
            array_expression = _pglast_parse_expression(expression["arg"],
                                                        available_tables=available_tables,
                                                        resolved_columns=resolved_columns, schema=schema)

            for index_expression in expression["indirection"]:
                index_expression: dict = index_expression["A_Indices"]

                if index_expression.get("is_slice", False):
                    lower = (_pglast_parse_expression(index_expression["lidx"],
                                                      available_tables=available_tables,
                                                      resolved_columns=resolved_columns, schema=schema)
                             if "lidx" in index_expression else None)
                    upper = (_pglast_parse_expression(index_expression["uidx"],
                                                      available_tables=available_tables,
                                                      resolved_columns=resolved_columns, schema=schema)
                             if "uidx" in index_expression else None)
                    array_expression = ArrayAccessExpression(array_expression, lower_idx=lower, upper_idx=upper)
                    continue

                point_index = _pglast_parse_expression(index_expression["uidx"], available_tables=available_tables,
                                                       resolved_columns=resolved_columns, schema=schema)
                array_expression = ArrayAccessExpression(array_expression, idx=point_index)

            return array_expression

        case _:
            raise ParserError("Unknown expression type: " + str(pglast_data))


def _pglast_parse_values_cte(pglast_data: dict) -> tuple[ValuesList, list[str]]:
    """Handler method to parse a CTE with a **VALUES** expressions.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the CTE data. This data is extracted from the pglast data structure.

    Returns
    -------
    tuple[ValuesList, list[str]]
        The parsed **VALUES** expression and the column names.
    """
    values: ValuesList = []
    for row in pglast_data["ctequery"]["SelectStmt"]["valuesLists"]:
        raw_items = row["List"]["items"]
        parsed_items = [_pglast_parse_expression(item, available_tables={}, resolved_columns={}, schema=None)
                        for item in raw_items]
        values.append(tuple(parsed_items))

    colnames: list[str] = []
    for raw_colname in pglast_data.get("aliascolnames", []):
        colnames.append(raw_colname["String"]["sval"])

    return values, colnames


def _pglast_parse_ctes(json_data: dict, *, available_tables: dict[str, TableReference],
                       resolved_columns: dict[tuple[str, str], ColumnReference],
                       schema: Optional["DatabaseSchema"]) -> CommonTableExpression:  # type: ignore # noqa: F821
    """Handler method to parse the **WITH** clause of a query.

    Parameters
    ----------
    json_data : dict
        JSON enconding of the CTEs, as extracted from the pglast data structure.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place as part of this method, in order to save the CTEs targets.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        This dictionary is not actually used in this method, but it is provided nonetheless to ensure a consistent API for
        all clause parsers.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

    Returns
    -------
    CommonTableExpression
        The parsed CTEs.
    """
    local_resolved_cols: dict[str, ColumnReference] = {}
    parsed_ctes: list[CommonTableExpression] = []
    for pglast_data in json_data["ctes"]:
        current_cte: dict = pglast_data["CommonTableExpr"]
        target_table = TableReference.create_virtual(current_cte["ctename"])
        available_tables[normalize(target_table.identifier())] = target_table

        match current_cte.get("ctematerialized", "CTEMaterializeDefault"):
            case "CTEMaterializeDefault":
                force_materialization = None
            case "CTEMaterializeAlways":
                force_materialization = True
            case "CTEMaterializeNever":
                force_materialization = False

        query_data = current_cte["ctequery"]["SelectStmt"]
        if "targetList" not in query_data and query_data["op"] == "SETOP_NONE":
            # CTE is a VALUES query
            values, columns = _pglast_parse_values_cte(current_cte)
            parsed_cte = ValuesWithQuery(values, target_name=target_table.identifier(),
                                         columns=columns, materialized=force_materialization)
        else:
            cte_query = _pglast_parse_query(current_cte["ctequery"]["SelectStmt"],
                                            available_tables=dict(available_tables),
                                            resolved_columns=local_resolved_cols,
                                            schema=schema)
            parsed_cte = WithQuery(cte_query, target_table, materialized=force_materialization)

        parsed_ctes.append(parsed_cte)

    recursive = json_data.get("recursive", False)

    return CommonTableExpression(parsed_ctes, recursive=recursive)


def _pglast_try_select_star(target: dict) -> Optional[Select]:
    """Attempts to generate a **SELECT(*)** representation for a **SELECT** clause.

    If the query is not actually a **SELECT(*)** query, this method will return **None**.

    Parameters
    ----------
    target : dict
        JSON encoding of the target entry in the **SELECT** clause. This data is extracted from the pglast data structure

    Returns
    -------
    Optional[Select]
        The parsed **SELECT(*)** clause, or **None** if this is not a **SELECT(*)** query.
    """
    if "ColumnRef" not in target:
        return None
    fields = target["ColumnRef"]["fields"]
    if len(fields) != 1:
        # multiple fields are used for qualified column references. This is definitely not a SELECT * query, so exit
        return None
    colref = fields[0]
    return Select.star() if "A_Star" in colref else None


def _pglast_parse_select(targetlist: list[dict], *, distinct: bool,
                         available_tables: dict[str, TableReference],
                         resolved_columns: dict[tuple[str, str], ColumnReference],
                         schema: Optional["DatabaseSchema"]) -> Select:  # type: ignore # noqa: F821
    """Handler method to parse the **SELECT** clause of a query.

    This is the only parsing handler that will always be called when parsing a query, since all queries must at least have a
    **SELECT** clause.

    Parameters
    ----------
    targetlist : list[dict]
        JSON encoding of the different projections used in the **SELECT** clause. This data is extracted from the pglast data
        structure.
    distinct : bool
        Whether this is a **SELECT DISTINCT** query.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

    Returns
    -------
    Select
        The parsed **SELECT** clause
    """
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
    """Handler method to extract the `TableReference` from a **RangeVar** entry in the **FROM** clause.

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
    """Checks, whether a pglast subquery representation refers to an actual subquery or a **VALUES** list.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the subquery data

    Returns
    -------
    bool
        **True** if the subquery encodes a **VALUES** list, **False** otherwise.
    """
    query = pglast_data["subquery"]["SelectStmt"]
    return "valuesLists" in query


def _pglast_parse_from_entry(pglast_data: dict, *, available_tables: dict[str, TableReference],
                             resolved_columns: dict[tuple[str, str], ColumnReference],
                             schema: Optional["DatabaseSchema"]) -> TableSource:  # type: ignore # noqa: F821
    """Handler method to parse individual entries in the **FROM** clause.

    Parameters
    ----------
    pglast_data : dict
        JSON enconding of the current entry in the **FROM** clause. This data is extracted from the pglast data structure.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place as part of this method.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place as part of this method.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

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
            similar_table = available_tables.get(normalize(table.full_name), None)
            if similar_table and similar_table.virtual and not table.alias:
                return DirectTableSource(similar_table)
            if similar_table and similar_table.virtual and table.alias:
                table = table.make_virtual()
                # TODO: should we also update the mapping of the full_name here?
                available_tables[normalize(table.alias)] = table
                return DirectTableSource(table)

            available_tables[normalize(table.full_name)] = table
            if table.alias:
                available_tables[normalize(table.alias)] = table

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

            left = _pglast_parse_from_entry(join_expr["larg"], available_tables=available_tables,
                                            resolved_columns=resolved_columns, schema=schema)
            right = _pglast_parse_from_entry(join_expr["rarg"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
            if join_type == JoinType.CrossJoin:
                return JoinTableSource(left, right, join_type=JoinType.CrossJoin)

            join_condition = _pglast_parse_predicate(join_expr["quals"], available_tables=available_tables,
                                                     resolved_columns=resolved_columns, schema=schema)

            # we do not need to store new tables in available_tables here, since this is already handled by the recursion.
            return JoinTableSource(left, right, join_condition=join_condition, join_type=join_type)

        case "RangeSubselect" if _pglast_is_values_list(pglast_data["RangeSubselect"]):
            values_list = _pglast_parse_values(pglast_data["RangeSubselect"])
            target_identifier = values_list.table.identifier() if values_list.table else ""
            if target_identifier:
                available_tables[normalize(values_list.table.identifier())] = values_list.table

            for target_column in values_list.cols:
                col_key = (target_identifier, target_column.name)
                resolved_columns[col_key] = target_column

            return values_list

        case "RangeSubselect":
            raw_subquery: dict = pglast_data["RangeSubselect"]
            is_lateral = raw_subquery.get("lateral", False)
            local_available_tables = dict(available_tables) if is_lateral else {}
            local_resolved_cols = dict(resolved_columns) if is_lateral else {}
            subquery = _pglast_parse_query(raw_subquery["subquery"]["SelectStmt"],
                                           available_tables=local_available_tables,
                                           resolved_columns=local_resolved_cols,
                                           schema=schema)

            if "alias" in raw_subquery:
                alias: str = raw_subquery["alias"]["aliasname"]
            else:
                alias = ""

            subquery_source = SubqueryTableSource(subquery, target_name=alias, lateral=is_lateral)
            if subquery_source.target_name:
                available_tables[normalize(subquery_source.target_table.identifier())] = subquery_source.target_table
            return subquery_source

        case _:
            raise ParserError("Unknow FROM clause entry: " + str(pglast_data))


def _pglast_parse_values(pglast_data: dict) -> ValuesTableSource:
    """Handler method to parse explicit **VALUES** lists in the **FROM** clause.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the actual **VALUES** list. This data is extracted from the pglast data structure and should be akin
        to a subquery.

    Returns
    -------
    ValuesTableSource
        The parsed **VALUES** list.
    """

    raw_values: list[dict] = pglast_data["subquery"]["SelectStmt"]["valuesLists"]

    values: ValuesList = []
    for row in raw_values:
        raw_items = row["List"]["items"]
        parsed_items = [_pglast_parse_expression(item, available_tables={}, resolved_columns={}, schema=None)
                        for item in raw_items]
        values.append(tuple(parsed_items))

    if "alias" not in pglast_data:
        return ValuesTableSource(values, alias="", columns=[])

    raw_alias = pglast_data["alias"]
    alias = raw_alias["aliasname"]
    if "colnames" not in raw_alias:
        return ValuesTableSource(values, alias=alias, columns=[])

    colnames = []
    for raw_colname in raw_alias["colnames"]:
        colnames.append(raw_colname["String"]["sval"])
    return ValuesTableSource(values, alias=alias, columns=colnames)


def _pglast_parse_from(from_clause: list[dict], *,
                       available_tables: dict[str, TableReference],
                       resolved_columns: dict[tuple[str, str], ColumnReference],
                       schema: Optional["DatabaseSchema"]) -> From:  # type: ignore # noqa: F821
    """Handler method to parse the **FROM** clause of a query.

    Parameters
    ----------
    from_clause : list[dict]
        The JSON representation of the **FROM** clause, as extracted from the pglast data structure.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

    Returns
    -------
    From
        The parsed **FROM** clause.
    """
    contains_plain_table = False
    contains_join = False
    contains_mixed = False  # plain tables and explicit JOINs, subqueries or VALUES

    table_sources: list[TableSource] = []
    for entry in from_clause:
        current_table_source = _pglast_parse_from_entry(entry, available_tables=available_tables,
                                                        resolved_columns=resolved_columns, schema=schema)
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


def _pglast_parse_predicate(pglast_data: dict, available_tables: dict[str, TableReference],
                            resolved_columns: dict[tuple[str, str], ColumnReference],
                            schema: Optional["DatabaseSchema"]) -> AbstractPredicate:  # type: ignore # noqa: F821
    """Handler method to parse arbitrary predicates in the **WHERE** or **HAVING** clause.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the predicate data. This data is extracted from the pglast data structure.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

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
            left = _pglast_parse_expression(expression["lexpr"], available_tables=available_tables,
                                            resolved_columns=resolved_columns, schema=schema)
            right = _pglast_parse_expression(expression["rexpr"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_LIKE":
            expression = pglast_data["A_Expr"]
            operator = (LogicalOperator.Like if expression["name"][0]["String"]["sval"] == "~~"
                        else LogicalOperator.NotLike)
            left = _pglast_parse_expression(expression["lexpr"], available_tables=available_tables,
                                            resolved_columns=resolved_columns, schema=schema)
            right = _pglast_parse_expression(expression["rexpr"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
            return BinaryPredicate(operator, left, right)

        case "A_Expr" if pglast_data["A_Expr"]["kind"] == "AEXPR_ILIKE":
            expression = pglast_data["A_Expr"]
            operator = (LogicalOperator.ILike if expression["name"][0]["String"]["sval"] == "~~*"
                        else LogicalOperator.NotILike)
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
            operator = _PglastOperatorMap[expression["boolop"]]
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
            operation = LogicalOperator.Is if expression["nulltesttype"] == "IS_NULL" else LogicalOperator.IsNot
            return BinaryPredicate(operation, testexpr, StaticValueExpression.null())

        case "FuncCall":
            expression = _pglast_parse_expression(pglast_data, available_tables=available_tables,
                                                  resolved_columns=resolved_columns, schema=schema)
            return UnaryPredicate(expression)

        case "SubLink":
            expression = pglast_data["SubLink"]
            sublink_type = expression["subLinkType"]

            subquery = _pglast_parse_query(expression["subselect"]["SelectStmt"],
                                           available_tables=dict(available_tables),
                                           resolved_columns=dict(resolved_columns),
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
            expression = _pglast_parse_expression(pglast_data, available_tables=available_tables,
                                                  resolved_columns=resolved_columns, schema=schema)
            return UnaryPredicate(expression)


def _pglast_parse_where(where_clause: dict, *,
                        available_tables: dict[str, TableReference],
                        resolved_columns: dict[tuple[str, str], ColumnReference],
                        schema: Optional["DatabaseSchema"]) -> Where:  # type: ignore # noqa: F821
    """Handler method to parse the **WHERE** clause of a query.

    Parameters
    ----------
    where_clause : dict
        The JSON representation of the **WHERE** clause, as extracted from the pglast data structure.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

    Returns
    -------
    Where
        The parsed **WHERE** clause.
    """
    predicate = _pglast_parse_predicate(where_clause, available_tables=available_tables,
                                        resolved_columns=resolved_columns, schema=schema)
    return Where(predicate)


def _pglast_parse_groupby(groupby_clause: list[dict], *,
                          available_tables: dict[str, TableReference],
                          resolved_columns: dict[tuple[str, str], ColumnReference],
                          schema: Optional["DatabaseSchema"]) -> GroupBy:  # type: ignore # noqa: F821
    """Handler method to parse the **GROUP BY** clause of a query.

    Parameters
    ----------
    groupby_clause : list[dict]
        The JSON representation of the **GROUP BY** clause, as extracted from the pglast data structure
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

    Returns
    -------
    GroupBy
        The parsed **GROUP BY** clause.
    """
    groupings: list[SqlExpression] = []

    for item in groupby_clause:
        if "GroupingSet" in item:
            raise NotImplementedError("Grouping sets are not yet supported")
        group_expression = _pglast_parse_expression(item, available_tables=available_tables,
                                                    resolved_columns=resolved_columns, schema=schema)
        groupings.append(group_expression)

    return GroupBy(groupings)


def _pglast_parse_having(having_clause: dict, *,
                         available_tables: dict[str, TableReference],
                         resolved_columns: dict[tuple[str, str], ColumnReference],
                         schema: Optional["DatabaseSchema"]) -> Having:  # type: ignore # noqa: F821
    """Handler method to parse the **HAVING** clause of a query.

    Parameters
    ----------
    having_clause : dict
        The JSON representation of the **HAVING** clause, as extracted from the pglast data structure.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

    Returns
    -------
    Having
        The parsed **HAVING** clause.
    """
    predicate = _pglast_parse_predicate(having_clause, available_tables=available_tables,
                                        resolved_columns=resolved_columns, schema=schema)
    return Having(predicate)


def _pglast_parse_orderby(order_clause: list[dict], *,
                          available_tables: dict[str, TableReference],
                          resolved_columns: dict[tuple[str, str], ColumnReference],
                          schema: Optional["DatabaseSchema"]) -> OrderBy:  # type: ignore # noqa: F821
    """Handler method to parse the **ORDER BY** clause of a query.

    Parameters
    ----------
    order_clause : list[dict]
        The JSON representation of the **ORDER BY** clause, as extracted from the pglast data structure.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

    Returns
    -------
    OrderBy
        The parsed **ORDER BY** clause.
    """
    orderings: list[OrderByExpression] = []

    for item in order_clause:
        expression = item["SortBy"]
        sort_key = _pglast_parse_expression(expression["node"], available_tables=available_tables,
                                            resolved_columns=resolved_columns, schema=schema)

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


def _pglast_parse_limit(pglast_data: dict, *, available_tables: dict[str, TableReference],
                        resolved_columns: dict[tuple[str, str], ColumnReference],
                        schema: Optional["DatabaseSchema"]) -> Optional[Limit]:  # type: ignore # noqa: F821
    """Handler method to parse LIMIT and OFFSET clauses.

    This method assumes that the given query actually contains **LIMIT** or **OFFSET** clauses and will fail otherwise.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the entire query. The method takes care of accessing the appropriate keys by itself, no preparation
        of the ``SelectStmt`` is necessary.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause. This is just provided to have a uniform interface for all parsing methods and not required for parsing of
        limits.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
        This is just provided to have a uniform interface for all parsing methods and not required for parsing of limits.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound. This is just provided to have a uniform interface for all parsing
        methods and not required for parsing of limits.

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


def _pglast_parse_setop(pglast_data: dict, *, available_tables: dict[str, TableReference],
                        resolved_columns: dict[tuple[str, str], ColumnReference],
                        schema: Optional["DatabaseSchema"]) -> SetQuery:  # type: ignore # noqa: F821
    """Handler method to parse set operations.

    This method assumes that the given query is indeed a set operation and will fail otherwise.

    Parameters
    ----------
    pglast_data : dict
        JSON encoding of the entire query. The method takes care of accessing the appropriate keys by itself, no preparation
        of the ``SelectStmt`` is necessary.
    available_tables : dict[str, TableReference]
        Candidate tables that columns can bind to. This dictionary maps the table identifier to the full table reference.
        An identifier can either be the full table name, or its alias.
        Note that this dictionary is modified in-place during the parsing process, especially while parsing CTEs and FROM
        clause.
    resolved_columns : dict[tuple[str, str], ColumnReference]
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
        Note that this dictionary is modified in-place during the parsing process while new columns are being discovered.
    schema : Optional[DatabaseSchema]
        The database schema to use for live binding of un-qualified column references. If this is omitted, no live binding is
        performed and some columns may remain unbound.

    Returns
    -------
    SetOperationClause
        The parsed set clause
    """
    if "withClause" in pglast_data:
        with_clause = _pglast_parse_ctes(pglast_data["withClause"],
                                         available_tables=available_tables,
                                         resolved_columns=resolved_columns, schema=schema)
    else:
        with_clause = None

    left_query = _pglast_parse_query(pglast_data["larg"],
                                     available_tables=dict(available_tables),
                                     resolved_columns=dict(resolved_columns),
                                     schema=schema)
    right_query = _pglast_parse_query(pglast_data["rarg"],
                                      available_tables=dict(available_tables),
                                      resolved_columns=dict(resolved_columns),
                                      schema=schema)

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
        order_clause = _pglast_parse_orderby(pglast_data["sortClause"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
    else:
        order_clause = None

    if pglast_data["limitOption"] == "LIMIT_OPTION_COUNT":
        limit_clause = _pglast_parse_limit(pglast_data, available_tables=available_tables,
                                           resolved_columns=resolved_columns, schema=schema)
    else:
        limit_clause = None

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


def _pglast_parse_query(stmt: dict, *, available_tables: dict[str, TableReference],
                        resolved_columns: dict[tuple[str, str], ColumnReference],
                        schema: Optional["DatabaseSchema"]) -> SelectStatement:  # type: ignore # noqa: F821
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
        Columns that have already been bound to their full column reference. This cache maps **(table, column)** pairs to
        their respective column objects. If columns do not use a qualified name, the **table** will be an empty string.
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
        return _pglast_parse_setop(stmt, available_tables=available_tables, resolved_columns=resolved_columns, schema=schema)

    clauses = []

    if "withClause" in stmt:
        with_clause = _pglast_parse_ctes(stmt["withClause"], available_tables=available_tables,
                                         resolved_columns=resolved_columns, schema=schema)
        clauses.append(with_clause)

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

    if "groupClause" in stmt:
        group_clause = _pglast_parse_groupby(stmt["groupClause"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
        clauses.append(group_clause)

    if "havingClause" in stmt:
        having_clause = _pglast_parse_having(stmt["havingClause"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
        clauses.append(having_clause)

    if "sortClause" in stmt:
        order_clause = _pglast_parse_orderby(stmt["sortClause"], available_tables=available_tables,
                                             resolved_columns=resolved_columns, schema=schema)
        clauses.append(order_clause)

    if stmt["limitOption"] == "LIMIT_OPTION_COUNT":
        limit_clause = _pglast_parse_limit(stmt, available_tables=available_tables,
                                           resolved_columns=resolved_columns, schema=schema)
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
    if parsed.is_set_query():
        return SetQuery(
            parsed.left_query, parsed.right_query,
            set_operation=parsed.set_operation,
            cte_clause=parsed.cte_clause,
            orderby_clause=parsed.orderby_clause,
            limit_clause=parsed.limit_clause,
            hint=hint, explain_clause=explain_clause
        )

    # must be a regular query then
    clauses = {
        "select_clause": parsed.select_clause,
        "from_clause": parsed.from_clause,
        "where_clause": parsed.where_clause,
        "groupby_clause": parsed.groupby_clause,
        "having_clause": parsed.having_clause,
        "orderby_clause": parsed.orderby_clause,
        "limit_clause": parsed.limit_clause,
        "cte_clause": parsed.cte_clause
    }

    if hint is not None:
        clauses["hints"] = hint
    if explain_clause is not None:
        clauses["explain_clause"] = explain_clause

    match parsed:
        case ImplicitSqlQuery():
            return ImplicitSqlQuery(**clauses)
        case ExplicitSqlQuery():
            return ExplicitSqlQuery(**clauses)
        case MixedSqlQuery():
            return MixedSqlQuery(**clauses)
        case SqlQuery():
            return SqlQuery(**clauses)
        case _:
            raise ValueError(f"Unknown query type: {parsed}")


@overload
def parse_query(query: str, *, include_hints: bool = True, bind_columns: bool | None = None,
                db_schema: Optional["DatabaseSchema"] = None) -> SqlQuery:  # type: ignore # noqa: F821
    ...


@overload
def parse_query(query: str, *, accept_set_query: bool, include_hints: bool = True,
                bind_columns: Optional[bool] = None,
                db_schema: Optional["DatabaseSchema"] = None) -> SelectStatement:  # type: ignore # noqa: F821
    ...


def parse_query(query: str, *, accept_set_query: bool = False, include_hints: bool = True,
                bind_columns: Optional[bool] = None,
                db_schema: Optional["DatabaseSchema"] = None) -> SelectStatement:  # type: ignore # noqa: F821
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
    db_schema : Optional[DatabaseSchema], optional
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

    parsed_query = _pglast_parse_query(stmt, available_tables={}, resolved_columns={}, schema=db_schema)
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
