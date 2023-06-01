"""The parser constructs `SqlQuery` objects from query strings.

Other than the parsing itself, the process will also execute a basic column binding process. For example, consider
a query like `SELECT * FROM R WHERE R.a = 42`. The binding affects the column reference `R.a` and sets the table of
that column to `R`. This binding based on column and table names is always performed.

If the table cannot be inferred based on the column name (e.g. for a query like `SELECT * FROM R WHERE a = 42`), a
second binding phase can be executed. This binding needs a working database connection and queries the database schema
to detect the correct tables for each column. Whether the second phase should also be executed by default can be
configured system-wide by setting the `auto_bind_columns` variable.
"""
from __future__ import annotations

import copy
import re
from typing import Any, Optional

import mo_sql_parsing as mosp

from postbound.qal import base, qal, clauses, expressions as expr, predicates as preds
from postbound.qal import transform
from postbound.db import db
from postbound.util import collections as collection_utils, dicts as dict_utils

auto_bind_columns: bool = False
"""Indicates whether the parser should use the database catalog to obtain column bindings."""

# The parser logic is based on the mo-sql-parsing project that implements a SQL -> JSON/dict conversion
# Our parser implementation takes such a JSON structure and constructs an equivalent SqlQuery object.
# The basic strategy for the parsing process is pretty straightforward: for each clause in the JSON data, there is
# a matching parsing method for our parser. This method then takes care of the appropriate conversion. For some parts,
# such as the parsing of predicates or expressions, more general methods exist that are shared by the clause parsing
# logic.

_MospSelectTypes = {
    "select": clauses.SelectType.Select,
    "select_distinct": clauses.SelectType.SelectDistinct
}

_MospJoinTypes = {  # see https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-JOIN
    # INNER JOIN
    "join": clauses.JoinType.InnerJoin,
    "inner join": clauses.JoinType.InnerJoin,

    # CROSS JOIN
    "cross join": clauses.JoinType.CrossJoin,

    # FULL OUTER JOIN
    "full join": clauses.JoinType.OuterJoin,
    "outer join": clauses.JoinType.OuterJoin,
    "full outer join": clauses.JoinType.OuterJoin,

    # LEFT OUTER JOIN
    "left join": clauses.JoinType.LeftJoin,
    "left outer join": clauses.JoinType.LeftJoin,

    # RIGHT OUTER JOIN
    "right join": clauses.JoinType.RightJoin,
    "right outer join": clauses.JoinType.RightJoin,

    # NATURAL INNER JOIN
    "natural join": clauses.JoinType.NaturalInnerJoin,
    "natural inner join": clauses.JoinType.NaturalInnerJoin,

    # NATURAL OUTER JOIN
    "natural outer join": clauses.JoinType.NaturalOuterJoin,
    "natural full outer join": clauses.JoinType.NaturalOuterJoin,

    # NATURAL LEFT OUTER JOIN
    "natural left join": clauses.JoinType.NaturalLeftJoin,
    "natural left outer join": clauses.JoinType.NaturalLeftJoin,

    # NATURAL RIGHT OUTER JOIN
    "natural right join": clauses.JoinType.NaturalRightJoin,
    "natural right outer join": clauses.JoinType.NaturalRightJoin
}

_MospAggregateOperations = {"count", "sum", "min", "max", "avg"}

_MospCompoundOperations = {
    "and": expr.LogicalSqlCompoundOperators.And,
    "or": expr.LogicalSqlCompoundOperators.Or,
    "not": expr.LogicalSqlCompoundOperators.Not
}

_MospUnaryOperations = {"exists": expr.LogicalSqlOperators.Exists, "missing": expr.LogicalSqlOperators.Missing}

_MospMathematicalOperations = {
    "add": expr.MathematicalSqlOperators.Add,
    "sub": expr.MathematicalSqlOperators.Subtract,
    "neg": expr.MathematicalSqlOperators.Negate,
    "mul": expr.MathematicalSqlOperators.Multiply,
    "div": expr.MathematicalSqlOperators.Divide,
    "mod": expr.MathematicalSqlOperators.Modulo
}

_MospBinaryOperations = {
    # comparison operators
    "eq": expr.LogicalSqlOperators.Equal,
    "neq": expr.LogicalSqlOperators.NotEqual,

    "lt": expr.LogicalSqlOperators.Less,
    "le": expr.LogicalSqlOperators.LessEqual,
    "lte": expr.LogicalSqlOperators.LessEqual,

    "gt": expr.LogicalSqlOperators.Greater,
    "ge": expr.LogicalSqlOperators.GreaterEqual,
    "gte": expr.LogicalSqlOperators.GreaterEqual,

    # other operators:
    "like": expr.LogicalSqlOperators.Like,
    "not_like": expr.LogicalSqlOperators.NotLike,
    "ilike": expr.LogicalSqlOperators.ILike,
    "not_ilike": expr.LogicalSqlOperators.NotILike,

    "in": expr.LogicalSqlOperators.In,
    "between": expr.LogicalSqlOperators.Between
}

_MospOperationSql = _MospCompoundOperations | _MospUnaryOperations | _MospMathematicalOperations | _MospBinaryOperations


def _parse_where_clause(mosp_data: dict) -> clauses.Where:
    if not isinstance(mosp_data, dict):
        raise ValueError("Unknown predicate format: " + str(mosp_data))
    return clauses.Where(_parse_mosp_predicate(mosp_data))


def _parse_mosp_predicate(mosp_data: dict) -> preds.AbstractPredicate:
    operation = dict_utils.key(mosp_data)

    # parse compound statements: AND / OR / NOT
    if operation in _MospCompoundOperations and operation != "not":
        child_statements = [_parse_mosp_predicate(child) for child in mosp_data[operation]]
        return preds.CompoundPredicate(_MospCompoundOperations[operation], child_statements)
    elif operation == "not":
        return preds.CompoundPredicate(_MospCompoundOperations[operation], _parse_mosp_predicate(mosp_data[operation]))

    # parse IS NULL / IS NOT NULL
    if operation in _MospUnaryOperations:
        return preds.UnaryPredicate(_parse_mosp_expression(mosp_data[operation]), _MospUnaryOperations[operation])

    # FIXME: cannot parse unary filter functions at the moment: SELECT * FROM R WHERE my_udf(R.a)
    # this likely requires changes to the UnaryPredicate implementation as well

    if operation not in _MospBinaryOperations:
        return preds.UnaryPredicate(_parse_mosp_expression(mosp_data))

    # parse binary predicates (logical operators, etc.)
    if operation == "in":
        return _parse_in_predicate(mosp_data)
    elif operation == "between":
        target_column, interval_start, interval_end = mosp_data[operation]
        parsed_column = _parse_mosp_expression(target_column)
        parsed_interval = (_parse_mosp_expression(interval_start), _parse_mosp_expression(interval_end))
        return preds.BetweenPredicate(parsed_column, parsed_interval)
    else:
        first_arg, second_arg = mosp_data[operation]
        return preds.BinaryPredicate(_MospOperationSql[operation], _parse_mosp_expression(first_arg),
                                     _parse_mosp_expression(second_arg))


def _parse_in_predicate(mosp_data: dict) -> preds.InPredicate:
    target_column, values = mosp_data["in"]
    parsed_column = _parse_mosp_expression(target_column)
    if isinstance(values, list):
        parsed_values = [_parse_mosp_expression(val) for val in values]
    elif isinstance(values, dict) and "literal" in values:
        # This weird wrap/unwrap logic is necessary b/c mosp _can_ return lists of string literals as
        # {"literal": ["a", "b", "c"]} instead of [{"literal": "a"}, {"literal": "b"}, {"literal": "c"]}], but the
        # parse_expression method only handles single expressions, not lists of them (i.e. produces one static value
        # expression in this case, rather than a list of expressions).
        # At the same time, an IN literal of a single value must be handled as well (e.g. IN ('foo')), which is parsed
        # by mosp as {"literal": "foo"} without any lists
        # Therefore we enlist the literal values first, and then construct individual literal clauses for each of them.
        parsed_values = [_parse_mosp_expression({"literal": val}) for val in collection_utils.enlist(values["literal"])]
    else:
        parsed_values = [_parse_mosp_expression(values)]
    return preds.InPredicate(parsed_column, parsed_values)


def _parse_mosp_expression(mosp_data: Any) -> expr.SqlExpression:
    # TODO: support for CASE WHEN expressions
    # TODO: support for string concatenation

    if mosp_data == "*":
        return expr.StarExpression()
    if isinstance(mosp_data, str):
        return expr.ColumnExpression(_parse_column_reference(mosp_data))

    if not isinstance(mosp_data, dict):
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
        return expr.MathematicalExpression(_MospOperationSql[operation], first_arg, remaining_args)

    # parse aggregate (COUNT / AVG / MIN / ...) or function call (CURRENT_DATE() etc)
    arguments = mosp_data[operation] if mosp_data[operation] else []
    if isinstance(arguments, list):
        parsed_arguments = [_parse_mosp_expression(arg) for arg in arguments]
    else:
        parsed_arguments = [_parse_mosp_expression(arguments)]
    return expr.FunctionExpression(operation, parsed_arguments, distinct=distinct)


def _parse_select_statement(mosp_data: dict | str) -> clauses.BaseProjection:
    if isinstance(mosp_data, dict):
        select_target = copy.copy(mosp_data["value"])
        target_name = mosp_data.get("name", None)
        return clauses.BaseProjection(_parse_mosp_expression(select_target), target_name)
    if mosp_data == "*":
        return clauses.BaseProjection.star()
    target_column = _parse_column_reference(mosp_data)
    return clauses.BaseProjection(expr.ColumnExpression(target_column))


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

    return clauses.Select(parsed_targets, _MospSelectTypes[select_type])


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
        return clauses.ImplicitFromClause(clauses.DirectTableSource(_parse_table_reference(from_clause)))
    elif isinstance(from_clause, dict):
        return clauses.ImplicitFromClause(clauses.DirectTableSource(_parse_table_reference(from_clause)))
    elif not isinstance(from_clause, list):
        raise TypeError("Unknown FROM clause structure: " + str(from_clause))
    parsed_sources = [clauses.DirectTableSource(_parse_table_reference(table)) for table in from_clause]
    return clauses.ImplicitFromClause(parsed_sources)


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
        join_type = next(jt for jt in _MospJoinTypes.keys() if jt in joined_table)
        parsed_join_type = _MospJoinTypes[join_type]
        join_source = joined_table[join_type]

        # TODO: enable USING support in addition to ON

        if (isinstance(join_source, dict)
                and ("select" in join_source["value"] or "select_distinct" in join_source["value"])):
            # we found a subquery
            joined_subquery = _MospQueryParser(join_source["value"], mosp.format(join_source)).parse_query()
            join_alias = joined_table.get("name", None)
            parsed_joins.append(clauses.JoinTableSource(clauses.SubqueryTableSource(joined_subquery, join_alias),
                                                        join_condition, join_type=parsed_join_type))
        elif isinstance(join_source, dict):
            # we found a normal table join with an alias
            table_name = join_source["value"]
            table_alias = join_source.get("name", None)
            table = base.TableReference(table_name, table_alias)
            parsed_joins.append(clauses.JoinTableSource(clauses.DirectTableSource(table), join_condition,
                                                        join_type=parsed_join_type))
        elif isinstance(join_source, str):
            # we found a normal table join without an alias
            table_name = join_source
            table = base.TableReference(table_name)
            parsed_joins.append(clauses.JoinTableSource(clauses.DirectTableSource(table), join_condition,
                                                        join_type=parsed_join_type))
        else:
            raise ValueError("Unknown JOIN format: " + str(joined_table))
    return clauses.ExplicitFromClause(clauses.DirectTableSource(initial_table), parsed_joins)


def _parse_base_table_source(mosp_data: dict) -> clauses.DirectTableSource | clauses.SubqueryTableSource:
    if isinstance(mosp_data, str):
        return clauses.DirectTableSource(_parse_table_reference(mosp_data))
    if not isinstance(mosp_data, dict) or "value" not in mosp_data or "name" not in mosp_data:
        raise TypeError("Unknown FROM clause target: " + str(mosp_data))

    value_target = mosp_data["value"]
    if isinstance(value_target, str):
        return clauses.DirectTableSource(_parse_table_reference(mosp_data))
    is_subquery_table = (isinstance(value_target, dict)
                         and any(select_type in value_target for select_type in _MospSelectTypes))
    if not is_subquery_table:
        raise TypeError("Unknown FROM clause target: " + str(mosp_data))
    parsed_subquery = _MospQueryParser(value_target, mosp.format(value_target)).parse_query()
    subquery_target = mosp_data["name"]
    return clauses.SubqueryTableSource(parsed_subquery, subquery_target)

def _parse_join_table_source(mosp_data: dict) -> clauses.JoinTableSource:
    join_type = next(jt for jt in _MospJoinTypes.keys() if jt in mosp_data)
    join_target = mosp_data[join_type]
    parsed_target = _parse_base_table_source(join_target)
    parsed_join_type = _MospJoinTypes[join_type]
    join_condition = _parse_mosp_predicate(mosp_data["on"]) if "on" in mosp_data else None
    return clauses.JoinTableSource(parsed_target, join_condition, join_type=parsed_join_type)


def _parsed_mixed_from_clause(mosp_data: dict) -> clauses.From:
    if "from" not in mosp_data:
        return clauses.From([])
    from_clause = mosp_data["from"]

    if isinstance(from_clause, str):
        return clauses.From(clauses.DirectTableSource(_parse_table_reference(from_clause)))
    elif isinstance(from_clause, dict):
        return clauses.From(_parse_base_table_source(from_clause))
    elif not isinstance(from_clause, list):
        raise TypeError("Unknown FROM clause type: " + str(from_clause))

    parsed_from_clause_entries = []
    for entry in from_clause:
        join_entry = isinstance(entry, dict) and any(join_type in entry for join_type in _MospJoinTypes)
        parsed_entry = _parse_join_table_source(entry) if join_entry else _parse_base_table_source(entry)
        parsed_from_clause_entries.append(parsed_entry)
    return clauses.From(parsed_from_clause_entries)


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
        groupby_clause = clauses.GroupBy(groupby_clause.group_columns, True)
        return groupby_clause
    else:
        columns = [_parse_mosp_expression(mosp_data)]
        return clauses.GroupBy(columns, False)


def _parse_having_clause(mosp_data: dict) -> clauses.Having:
    return clauses.Having(_parse_mosp_predicate(mosp_data))


def _parse_orderby_expression(mosp_data: dict) -> clauses.OrderByExpression:
    column = _parse_mosp_expression(mosp_data["value"])
    ascending = mosp_data["sort"] == "asc" if "sort" in mosp_data else None
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
        if not isinstance(table, dict):
            continue
        explicit_join = any(join_type in table for join_type in _MospJoinTypes)
        subquery_table = any(select_type in table for select_type in _MospSelectTypes)
        nested_subquery = "value" in table and any(select_type in table["value"] for select_type in _MospSelectTypes)
        if explicit_join or subquery_table or nested_subquery:
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
        explicit_join = isinstance(table, dict) and any(join_type in table for join_type in _MospJoinTypes)
        subquery_table = (isinstance(table, dict)
                          and (any(select_type in table for select_type in _MospSelectTypes) or "value" in table))
        if not explicit_join or subquery_table:
            return False
    return True


class QueryFormatError(RuntimeError):
    def __init__(self, query: str) -> None:
        super().__init__(f"Query must be either explicit or implicit, not a mixture of both: '{query}'")
        self.query = query


class _MospQueryParser:
    """The parser class acts as a one-stop-shop to parse the input query."""

    def __init__(self, mosp_data: dict, raw_query: str = "") -> None:
        self._raw_query = raw_query
        self._mosp_data = mosp_data
        self._explain = {}

        self._prepare_query()

    def parse_query(self) -> qal.SqlQuery:
        if _is_implicit_query(self._mosp_data):
            implicit, explicit = True, False
            from_clause = _parse_implicit_from_clause(self._mosp_data)
        elif _is_explicit_query(self._mosp_data):
            implicit, explicit = False, True
            from_clause = _parse_explicit_from_clause(self._mosp_data)
        else:
            implicit, explicit = False, False
            from_clause = _parsed_mixed_from_clause(self._mosp_data)
        select_clause = _parse_select_clause(self._mosp_data)
        where_clause = _parse_where_clause(self._mosp_data["where"]) if "where" in self._mosp_data else None

        # TODO: support for EXPLAIN queries
        # TODO: support for CTEs

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

        if implicit and not explicit:
            return qal.ImplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                        groupby_clause=groupby_clause, having_clause=having_clause,
                                        orderby_clause=orderby_clause, limit_clause=limit_clause)
        elif not implicit and explicit:
            return qal.ExplicitSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                        groupby_clause=groupby_clause, having_clause=having_clause,
                                        orderby_clause=orderby_clause, limit_clause=limit_clause)
        else:
            return qal.MixedSqlQuery(select_clause=select_clause, from_clause=from_clause, where_clause=where_clause,
                                     groupby_clause=groupby_clause, having_clause=having_clause,
                                     orderby_clause=orderby_clause, limit_clause=limit_clause)

    def _prepare_query(self) -> None:
        if "explain" in self._mosp_data:
            self._explain = {"analyze": self._mosp_data.get("analyze", False),
                             "format": self._mosp_data.get("format", "text")}
            self._mosp_data = self._mosp_data["explain"]


def parse_query(query: str, *, bind_columns: bool | None = None,
                db_schema: Optional[db.DatabaseSchema] = None) -> qal.SqlQuery:
    """Parses the given query string into a `SqlQuery` object.

    If `bind_columns` is `True`, will perform a binding process based on the schema of a live database.
    This database schema can be either supplied directly via the `db_schema` parameter, otherwise it will be fetched
    from the `DatabasePool`.

    If `bind_columns` is omitted, the `auto_bind_columns` variable will be queried.
    """
    bind_columns = bind_columns if bind_columns is not None else auto_bind_columns
    db_schema = (db_schema if db_schema or not bind_columns
                 else db.DatabasePool.get_instance().current_database().schema())
    mosp_data = mosp.parse(query)
    parsed_query = _MospQueryParser(mosp_data, query).parse_query()
    return transform.bind_columns(parsed_query, with_schema=bind_columns, db_schema=db_schema)
