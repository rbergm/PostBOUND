from __future__ import annotations

import copy
import re
from typing import Any

import mo_sql_parsing as mosp

import postbound.qal.clauses
from postbound.qal import base, clauses, expressions as expr, joins, qal, predicates as preds
from postbound.qal import transform
from postbound.db import db
from postbound.util import dicts as dict_utils

AutoBindColumns: bool = False

_MospCompoundOperations = {"and", "or", "not"}
_MospBinaryOperations = {"lt", "gt", "le", "ge", "gte", "lte", "eq", "neq", "like", "not_like", "ilike", "not_ilike",
                         "in", "between"}
_MospUnaryOperations = {"exists", "missing"}
_MospAggregateOperations = {"count", "sum", "min", "max", "avg"}
_MospMathematicalOperations = {"add", "sub", "neg", "mul", "div", "mod", "and", "or", "not"}
_MospJoinTypes = {"join", "cross join", "full join", "left join", "right join", "outer join", "inner join",
                  "natural join", "left outer join", "right outer join", "full outer join"}


def _parse_where_clause(mosp_data: dict) -> clauses.Where:
    if not isinstance(mosp_data, dict):
        raise ValueError("Unknown predicate format: " + str(mosp_data))
    return clauses.Where(_parse_mosp_predicate(mosp_data))


def _parse_mosp_predicate(mosp_data: dict) -> preds.AbstractPredicate:
    operation = dict_utils.key(mosp_data)

    # parse compound statements: AND / OR / NOT
    if operation in _MospCompoundOperations and operation != "not":
        child_statements = [_parse_mosp_predicate(child) for child in mosp_data[operation]]
        return preds.CompoundPredicate(operation, child_statements)
    elif operation == "not":
        return preds.CompoundPredicate(operation, _parse_mosp_predicate(mosp_data[operation]))

    # parse IS NULL / IS NOT NULL
    if operation in _MospUnaryOperations:
        return preds.UnaryPredicate(operation, _parse_mosp_expression(mosp_data[operation]), mosp_data)

    if operation not in _MospBinaryOperations:
        raise ValueError("Unknown predicate format: " + str(mosp_data))

    # parse binary predicates (logical operators, etc.)
    if operation == "in":
        target_column, *values = mosp_data[operation]
        parsed_column = _parse_mosp_expression(target_column)
        if len(values) == 1 and isinstance(values[0], dict) and "literal" in values[0]:
            parsed_values = [expr.StaticValueExpression(val) for val in values[0]["literal"]]
        else:
            parsed_values = [_parse_mosp_expression(val) for val in values]
        return preds.InPredicate(parsed_column, parsed_values, mosp_data)
    elif operation == "between":
        target_column, interval_start, interval_end = mosp_data[operation]
        parsed_column = _parse_mosp_expression(target_column)
        parsed_interval = (_parse_mosp_expression(interval_start), _parse_mosp_expression(interval_end))
        return preds.BetweenPredicate(parsed_column, parsed_interval, mosp_data)
    else:
        first_arg, second_arg = mosp_data[operation]
        return preds.BinaryPredicate(operation, _parse_mosp_expression(first_arg),
                                     _parse_mosp_expression(second_arg), mosp_data)


def _parse_mosp_expression(mosp_data: Any) -> expr.SqlExpression:
    if mosp_data == "*":
        return expr.StarExpression()
    if isinstance(mosp_data, str):
        return expr.ColumnExpression(_parse_column_reference(mosp_data))
    elif not isinstance(mosp_data, dict):
        return expr.StaticValueExpression(mosp_data)

    # parse string literals
    if "literal" in mosp_data:
        return expr.StaticValueExpression(mosp_data["literal"])

    # parse subqueries
    if "select" in mosp_data or "select_distinct" in mosp_data:
        subquery = _MospQueryParser(mosp_data, mosp.format(mosp_data)).parse_query()
        return expr.SubqueryExpression(subquery)

    # parse value CASTs and mathematical operations (+ / - etc), including logical operations
    mosp_data: dict = copy.copy(mosp_data)
    distinct = mosp_data.pop("distinct") if "distinct" in mosp_data else None  # side effect is intentional!
    operation = dict_utils.key(mosp_data)
    if operation == "cast":
        cast_target, cast_type = mosp_data["cast"]
        return expr.CastExpression(_parse_mosp_expression(cast_target), dict_utils.key(cast_type))
    elif operation in _MospMathematicalOperations:
        parsed_arguments = [_parse_mosp_expression(arg) for arg in mosp_data[operation]]
        first_arg, *remaining_args = parsed_arguments
        return expr.MathematicalExpression(operation, first_arg, remaining_args)

    # parse aggregate (COUNT / AVG / MIN / ...) or function call (CURRENT_DATE() etc)
    arguments = mosp_data[operation] if mosp_data[operation] else []
    if isinstance(arguments, list):
        parsed_arguments = [_parse_mosp_expression(arg) for arg in arguments]
    else:
        parsed_arguments = [_parse_mosp_expression(arguments)]
    return expr.FunctionExpression(operation, parsed_arguments, distinct=distinct)


def _parse_select_statement(mosp_data: dict | str) -> postbound.qal.clauses.BaseProjection:
    if isinstance(mosp_data, dict):
        select_target = copy.copy(mosp_data["value"])
        target_name = mosp_data.get("name", None)
        return postbound.qal.clauses.BaseProjection(_parse_mosp_expression(select_target), target_name)
    target_column = _parse_column_reference(mosp_data)
    return postbound.qal.clauses.BaseProjection(expr.ColumnExpression(target_column))


def _parse_select_clause(mosp_data: dict) -> clauses.Select:
    if "select" not in mosp_data and "select_distinct" not in mosp_data:
        raise ValueError("Unknown SELECT format: " + str(mosp_data))

    select_type = "select" if "select" in mosp_data else "select_distinct"
    select_targets = mosp_data[select_type]

    if isinstance(select_targets, list):
        parsed_targets = [_parse_select_statement(target) for target in select_targets]
    elif isinstance(select_targets, dict) or isinstance(select_targets, str):
        parsed_targets = [_parse_select_statement(select_targets)]
    else:
        raise ValueError("Unknown SELECT format: " + str(select_targets))

    return clauses.Select(parsed_targets, select_type)


# see https://regex101.com/r/HdKzQg/2
_TableReferencePattern = re.compile(r"(?P<full_name>\S+)( (AS )?(?P<alias>\S+))?")


def _parse_table_reference(table: str | dict) -> base.TableReference:
    # mo_sql table format
    if isinstance(table, dict):
        table_name = table["value"]
        table_alias = table.get("name", None)
        return base.TableReference(table_name, table_alias)

    # string-based table format
    pattern_match = _TableReferencePattern.match(table)
    if not pattern_match:
        raise ValueError(f"Could not parse table reference for '{table}'")
    full_name, alias = pattern_match.group("full_name", "alias")
    alias = "" if not alias else alias
    return base.TableReference(full_name, alias)


def _parse_column_reference(column: str) -> base.ColumnReference:
    if "." not in column:
        return base.ColumnReference(column)

    table, column = column.split(".")
    table_ref = base.TableReference("", table)
    return base.ColumnReference(column, table_ref)


def _parse_implicit_from_clause(mosp_data: dict) -> clauses.ImplicitFromClause:
    if "from" not in mosp_data:
        return clauses.ImplicitFromClause()
    from_clause = mosp_data["from"]
    if isinstance(from_clause, str):
        return clauses.ImplicitFromClause(_parse_table_reference(from_clause))
    return clauses.ImplicitFromClause([_parse_table_reference(table) for table in from_clause])


def _parse_explicit_from_clause(mosp_data: dict) -> clauses.ExplicitFromClause:
    if "from" not in mosp_data:
        raise ValueError("No tables in FROM clause")
    from_clause = mosp_data["from"]
    if not isinstance(from_clause, list):
        raise ValueError("Unknown FROM clause format: " + str(from_clause))
    first_table, *joined_tables = from_clause
    initial_table = _parse_table_reference(first_table)
    parsed_joins = []
    for joined_table in joined_tables:
        join_condition = _parse_mosp_predicate(joined_table["on"]) if "on" in joined_table else None
        join_type = next(jt for jt in _MospJoinTypes if jt in joined_table)
        join_source = joined_table[join_type]

        # TODO: enable USING support in addition to ON

        if (isinstance(join_source, dict)
                and ("select" in join_source["value"] or "select_distinct" in join_source["value"])):
            # we found a subquery
            joined_subquery = _MospQueryParser(join_source["value"], mosp.format(join_source)).parse_query()
            join_alias = joined_table.get("name", None)
            parsed_joins.append(joins.SubqueryJoin(join_type, joined_subquery, join_alias, join_condition))
        elif isinstance(join_source, dict):
            # we found a normal table join with an alias
            table_name = join_source["value"]
            table_alias = join_source.get("name", None)
            table = base.TableReference(table_name, table_alias)
            parsed_joins.append(joins.TableJoin(join_type, table, join_condition))
        elif isinstance(join_source, str):
            # we found a normal table join without an alias
            table_name = join_source
            table = base.TableReference(table_name)
            parsed_joins.append(joins.TableJoin(join_type, table, join_condition))
        else:
            raise ValueError("Unknown JOIN format: " + str(joined_table))
    return clauses.ExplicitFromClause(initial_table, parsed_joins)


def _parse_groupby_clause(mosp_data: dict | list) -> clauses.GroupBy:
    # The format of GROUP BY clauses is a bit weird in mo-sql. Therefore, the parsing logic looks quite hacky
    # Take a look at the MoSQLParsingTests for details.

    if isinstance(mosp_data, list):
        columns = [_parse_mosp_expression(col["value"]) for col in mosp_data]
        distinct = False
        return clauses.GroupBy(columns, distinct)

    mosp_data = mosp_data["value"]
    if "distinct" in mosp_data:
        groupby_clause = _parse_groupby_clause(mosp_data["distinct"])
        groupby_clause.distinct = True
        return groupby_clause
    else:
        columns = [_parse_mosp_expression(mosp_data)]
        distinct = False
        return clauses.GroupBy(columns, distinct)


def _parse_having_clause(mosp_data: dict) -> clauses.Having:
    return clauses.Having(_parse_mosp_predicate(mosp_data))


def _parse_orderby_expression(mosp_data: dict) -> clauses.OrderByExpression:
    column = _parse_mosp_expression(mosp_data["value"])
    ascending = mosp_data.get("sort", "asc") == "asc"
    return clauses.OrderByExpression(column, ascending)


def _parse_orderby_clause(mosp_data: dict | list) -> clauses.OrderBy:
    if isinstance(mosp_data, list):
        order_expressions = [_parse_orderby_expression(order_expr) for order_expr in mosp_data]
    else:
        order_expressions = [_parse_orderby_expression(mosp_data)]
    return clauses.OrderBy(order_expressions)


def _parse_limit_clause(mosp_data: dict) -> clauses.Limit:
    limit = mosp_data.get("limit", None)
    offset = mosp_data.get("offset", None)
    return clauses.Limit(limit=limit, offset=offset)


def _is_implicit_query(mosp_data: dict) -> bool:
    if "from" not in mosp_data:
        return True

    from_clause = mosp_data["from"]
    if not isinstance(from_clause, list):
        return True

    for table in from_clause:
        if isinstance(table, dict) and any(join_type in table for join_type in _MospJoinTypes):
            return False
    return True


def _is_explicit_query(mosp_data: dict) -> bool:
    if "from" not in mosp_data:
        return False

    from_clause = mosp_data["from"]
    if not isinstance(from_clause, list):
        return False

    base_table, *join_statements = from_clause

    for table in join_statements:
        if isinstance(table, str):
            return False
        if isinstance(table, dict) and not any(join_type in table for join_type in _MospJoinTypes):
            return False
    return True


class QueryFormatError(RuntimeError):
    def __init__(self, query: str) -> None:
        super().__init__(f"Query must be either explicit or implicit, not a mixture of both: '{query}'")
        self.query = query


class _MospQueryParser:
    def __init__(self, mosp_data: dict, raw_query: str = "") -> None:
        self._raw_query = raw_query
        self._mosp_data = mosp_data
        self._explain = {}

        self._prepare_query()

    def parse_query(self) -> qal.SqlQuery:
        if _is_implicit_query(self._mosp_data):
            implicit = True
            from_clause = _parse_implicit_from_clause(self._mosp_data)
        elif _is_explicit_query(self._mosp_data):
            implicit = False
            from_clause = _parse_explicit_from_clause(self._mosp_data)
        else:
            raise QueryFormatError(self._raw_query)
        select_clause = _parse_select_clause(self._mosp_data)
        where_clause = _parse_where_clause(self._mosp_data["where"]) if "where" in self._mosp_data else None

        if "groupby" in self._mosp_data:
            groupby_clause = _parse_groupby_clause(self._mosp_data["groupby"])
        else:
            groupby_clause = None

        if "having" in self._mosp_data:
            having_clause = _parse_having_clause(self._mosp_data["having"])
        else:
            having_clause = None

        if "orderby" in self._mosp_data:
            orderby_clause = _parse_orderby_clause(self._mosp_data["orderby"])
        else:
            orderby_clause = None

        if "limit" in self._mosp_data or "offset" in self._mosp_data:
            # LIMIT and OFFSET are both in mosp_data, no indirection necessary
            limit_clause = _parse_limit_clause(self._mosp_data)
        else:
            limit_clause = None

        if implicit:
            return qal.ImplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                        groupby_clause=groupby_clause, having_clause=having_clause,
                                        orderby_clause=orderby_clause, limit_clause=limit_clause)
        else:
            return qal.ExplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                        groupby_clause=groupby_clause, having_clause=having_clause,
                                        orderby_clause=orderby_clause, limit_clause=limit_clause)

    def _prepare_query(self) -> None:
        if "explain" in self._mosp_data:
            self._explain = {"analyze": self._mosp_data.get("analyze", False),
                             "format": self._mosp_data.get("format", "text")}
            self._mosp_data = self._mosp_data["explain"]


def parse_query(query: str, *,
                bind_columns: bool = AutoBindColumns, db_schema: db.DatabaseSchema | None = None) -> qal.SqlQuery:
    db_schema = db_schema if db_schema or not bind_columns else db.DatabasePool.get_instance().current_database()
    mosp_data = mosp.parse(query)
    parsed_query = _MospQueryParser(mosp_data, query).parse_query()
    transform.bind_columns(parsed_query, with_schema=bind_columns, db_schema=db_schema)
    return parsed_query
