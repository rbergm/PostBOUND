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
import re
from typing import Any, Optional

import mo_sql_parsing as mosp

from . import transform
from ._core import (
    TableReference, ColumnReference,
    SelectType, JoinType,
    CompoundOperators, MathematicalSqlOperators, LogicalSqlOperators,
    BaseProjection, WithQuery, DirectTableSource, JoinTableSource, SubqueryTableSource, OrderByExpression,
    SqlQuery, ImplicitSqlQuery, ExplicitSqlQuery, MixedSqlQuery,
    Select, Where, From, GroupBy, Having, OrderBy, Limit, CommonTableExpression, ImplicitFromClause, ExplicitFromClause,
    AbstractPredicate, BinaryPredicate, InPredicate, BetweenPredicate, CompoundPredicate, UnaryPredicate,
    SqlExpression, StarExpression, StaticValueExpression, ColumnExpression, SubqueryExpression, CastExpression,
    MathematicalExpression, FunctionExpression, BooleanExpression, WindowExpression, CaseExpression
)
from .transform import QueryType
from .. import util

auto_bind_columns: bool = False
"""Indicates whether the parser should use the database catalog to obtain column bindings."""

# The parser logic is based on the mo-sql-parsing project that implements a SQL -> JSON/dict conversion
# Our parser implementation takes such a JSON structure and constructs an equivalent SqlQuery object.
# The basic strategy for the parsing process is pretty straightforward: for each clause in the JSON data, there is
# a matching parsing method for our parser. This method then takes care of the appropriate conversion. For some parts,
# such as the parsing of predicates or expressions, more general methods exist that are shared by the clause parsing
# logic.
# In the course of our implementation, we will use mo-sql and mosp as shorthands to refer to mo-sql-parsing related
# functionality.
# A central problem when working with mo-sql is the very loose output format. Depending on the specific input query,
# at the same point in the dictionary very different value types can be contained. Furthermore, the dictionaries can
# also have different keys. This leads to a lot of "exploratory" coding and black-box testing which structures are
# produced under which conditions. Therefore, our high-level parser basically needs to reverse-engineer a portion of
# the original low-level parser. The conditions we test are document in the analysis/MoSQLParsingTests.ipynb Jupyter
# notebook. Still, there might be corner cases where our parser does not access the correct data or interprets it in
# a wrong way. The hope is that with enough time and testing, those cases will be identified and fixed.

_MospSelectTypes = {
    "select": SelectType.Select,
    "select_distinct": SelectType.SelectDistinct
}
"""The different kinds of ``SELECT`` clauses that mo-sql can emit"""

_MospJoinTypes = {
    # INNER JOIN
    "join": JoinType.InnerJoin,
    "inner join": JoinType.InnerJoin,

    # CROSS JOIN
    "cross join": JoinType.CrossJoin,

    # FULL OUTER JOIN
    "full join": JoinType.OuterJoin,
    "outer join": JoinType.OuterJoin,
    "full outer join": JoinType.OuterJoin,

    # LEFT OUTER JOIN
    "left join": JoinType.LeftJoin,
    "left outer join": JoinType.LeftJoin,

    # RIGHT OUTER JOIN
    "right join": JoinType.RightJoin,
    "right outer join": JoinType.RightJoin,

    # NATURAL INNER JOIN
    "natural join": JoinType.NaturalInnerJoin,
    "natural inner join": JoinType.NaturalInnerJoin,

    # NATURAL OUTER JOIN
    "natural outer join": JoinType.NaturalOuterJoin,
    "natural full outer join": JoinType.NaturalOuterJoin,

    # NATURAL LEFT OUTER JOIN
    "natural left join": JoinType.NaturalLeftJoin,
    "natural left outer join": JoinType.NaturalLeftJoin,

    # NATURAL RIGHT OUTER JOIN
    "natural right join": JoinType.NaturalRightJoin,
    "natural right outer join": JoinType.NaturalRightJoin
}
"""The different kinds of ``JOIN`` statements that mo-sql can emit, as well as their mapping to our counterparts.

See Also
--------
.. Postgres documentation of the various join types:
   https://www.postgresql.org/docs/current/queries-table-expressions.html#QUERIES-JOIN
"""

_MospCompoundOperations = {
    "and": CompoundOperators.And,
    "or": CompoundOperators.Or,
    "not": CompoundOperators.Not
}
"""The different kinds of aggregate operators that mo-sql can emit, as well as their mapping to our counterparts."""

_MospUnaryOperations = {"exists": LogicalSqlOperators.Exists, "missing": LogicalSqlOperators.Missing}
"""The different kinds of unary operators that mo-sql can emit, as well as their mapping to our counterparts."""

_MospMathematicalOperations = {
    "add": MathematicalSqlOperators.Add,
    "sub": MathematicalSqlOperators.Subtract,
    "neg": MathematicalSqlOperators.Negate,
    "mul": MathematicalSqlOperators.Multiply,
    "div": MathematicalSqlOperators.Divide,
    "mod": MathematicalSqlOperators.Modulo
}
"""The different kinds of mathematical operators that mo-sql can emit, as well as their mapping to our counterparts."""

_MospBinaryOperations = {
    # comparison operators
    "eq": LogicalSqlOperators.Equal,
    "neq": LogicalSqlOperators.NotEqual,

    "lt": LogicalSqlOperators.Less,
    "le": LogicalSqlOperators.LessEqual,
    "lte": LogicalSqlOperators.LessEqual,

    "gt": LogicalSqlOperators.Greater,
    "ge": LogicalSqlOperators.GreaterEqual,
    "gte": LogicalSqlOperators.GreaterEqual,

    # other operators:
    "like": LogicalSqlOperators.Like,
    "not_like": LogicalSqlOperators.NotLike,
    "ilike": LogicalSqlOperators.ILike,
    "not_ilike": LogicalSqlOperators.NotILike,

    "in": LogicalSqlOperators.In,
    "nin": LogicalSqlOperators.In,
    "between": LogicalSqlOperators.Between
}
"""The different kinds of mathematical operators that mo-sql can emit, as well as their mapping to our counterparts."""

_MospOperationSql = (_MospCompoundOperations
                     | _MospUnaryOperations
                     | _MospMathematicalOperations
                     | _MospBinaryOperations)
"""All different operators that mo-sql can emit as one large dictionary."""


def _parse_where_clause(mosp_data: dict) -> Where:
    """Parsing logic for the ``WHERE`` clause.

    Parameters
    ----------
    mosp_data : dict
        The mo-sql contents of the clause. This is the value of the ``where`` key.

    Returns
    -------
    Where
        The parsed clause

    Raises
    ------
    ValueError
        If the contents are not a dictionary. We never encountered this case during testing. If it can be emitted for
        a valid query, we need to extend our parsing logic.
    """
    if not isinstance(mosp_data, dict):
        raise ValueError("Unknown predicate format: " + str(mosp_data))
    return Where(_parse_mosp_predicate(mosp_data))


def _parse_mosp_predicate(mosp_data: dict) -> AbstractPredicate:
    """Parsing logic for arbitrary SQL predicates

    Parameters
    ----------
    mosp_data : dict
        The mo-sql contents of the predicate. This is a dictionary mapping a single key (the predicate's operator) to
        the contents.

    Returns
    -------
    AbstractPredicate
        The parsed predicate

    Raises
    ------
    ValueError
        If `mosp_data` contains multiple keys. In this case we cannot determine which is the operator and which is some
        other data, yet. However, we never encountered this case during testing. If it can be emitted for a valid
        query, we need to extend our parsing logic.
    """
    operation = util.dicts.key(mosp_data)

    # parse compound statements: AND / OR / NOT
    if operation in _MospCompoundOperations and operation != "not":
        child_statements = [_parse_mosp_predicate(child) for child in mosp_data[operation]]
        return CompoundPredicate(_MospCompoundOperations[operation], child_statements)
    elif operation == "not":
        return CompoundPredicate(_MospCompoundOperations[operation], _parse_mosp_predicate(mosp_data[operation]))

    # parse IS NULL / IS NOT NULL
    if operation in _MospUnaryOperations:
        return UnaryPredicate(_parse_mosp_expression(mosp_data[operation]), _MospUnaryOperations[operation])

    # FIXME: cannot parse unary filter functions at the moment: SELECT * FROM R WHERE my_udf(R.a)
    # this likely requires changes to the UnaryPredicate implementation as well
    # TODO: check if this is still valid. Most likely this has already been fixed and the comment was forgotten.

    if operation not in _MospBinaryOperations:
        return UnaryPredicate(_parse_mosp_expression(mosp_data))

    # parse binary predicates (logical operators, etc.)
    if operation == "in":
        return _parse_in_predicate(mosp_data)
    elif operation == "nin":
        mosp_data = copy.copy(mosp_data)
        mosp_data["in"] = mosp_data.pop("nin")
        in_predicate = _parse_in_predicate(mosp_data)
        return CompoundPredicate.create_not(in_predicate)
    elif operation == "between":
        target_column, interval_start, interval_end = mosp_data[operation]
        parsed_column = _parse_mosp_expression(target_column)
        parsed_interval = (_parse_mosp_expression(interval_start), _parse_mosp_expression(interval_end))
        return BetweenPredicate(parsed_column, parsed_interval)
    else:
        first_arg, second_arg = mosp_data[operation]
        return BinaryPredicate(_MospOperationSql[operation], _parse_mosp_expression(first_arg),
                               _parse_mosp_expression(second_arg))


def _parse_in_predicate(mosp_data: dict) -> InPredicate:
    """Parsing logic for ``IN`` predicates.

    Parameters
    ----------
    mosp_data : dict
        The mo-sql contents of the predicate. This is a dictionary mapping a single literal ``"in"`` key to the
        arguments that form the predicate. The first argument is the expression that should be any of the values of the
        second argument. Take a look at the implementation for more details on the precise layout the dictionary

    Returns
    -------
    InPredicate
        The parsed ``IN`` predicate.
    """
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
        parsed_values = [_parse_mosp_expression({"literal": val})
                         for val in util.enlist(values["literal"])]
    else:
        parsed_values = [_parse_mosp_expression(values)]
    return InPredicate(parsed_column, parsed_values)


def _parse_mosp_expression(mosp_data: Any) -> SqlExpression:
    """Parsing logic for arbitrary expressions.

    This method detects the actual expression type and delegates to a specialized parser if necessary. Consult the
    actual implementation to figure out which specific detection rules and heuristics are applied.

    Parameters
    ----------
    mosp_data : Any
        The complete mo-sql data that is used to describe the expression

    Returns
    -------
    SqlExpression
        The parsed expression
    """
    # TODO: support for CASE WHEN expressions
    # TODO: support for string concatenation

    if mosp_data == "*":
        return StarExpression()
    if isinstance(mosp_data, str):
        return ColumnExpression(_parse_column_reference(mosp_data))

    if not isinstance(mosp_data, dict):
        return StaticValueExpression(mosp_data)
    if "null" in mosp_data:
        # NULL values are encoded as {'null': {}} in mosql
        return StaticValueExpression(None)

    # parse string literals
    if "literal" in mosp_data:
        return StaticValueExpression(mosp_data["literal"])

    # parse subqueries
    if "select" in mosp_data or "select_distinct" in mosp_data:
        subquery = _MospQueryParser(mosp_data, mosp.format(mosp_data)).parse_query()
        return SubqueryExpression(subquery)

    if "over" in mosp_data:
        return _parse_window_function(mosp_data)
    if "case" in mosp_data:
        return _parse_case_expression(mosp_data["case"])

    # parse value CASTs and mathematical operations (+ / - etc), including logical operations

    # We need to copy the mosp_data here because we are going to modify it next. This prevents unintended side effects
    # on the supplied mosp_data argument
    mosp_data: dict = copy.copy(mosp_data)

    # We mutate our dictionary now in order to remove additional DISTINCT clause information.
    # If we leave this information in, we cannot use dict_utils.key to determine the operation since there would be
    # multiple keys left.
    distinct = mosp_data.pop("distinct") if "distinct" in mosp_data else None  # side effect is intentional!

    operation = util.dicts.key(mosp_data)
    if operation == "cast":
        cast_target, cast_type = mosp_data["cast"]
        return CastExpression(_parse_mosp_expression(cast_target), util.dicts.key(cast_type))
    elif operation in _MospMathematicalOperations:
        parsed_arguments = [_parse_mosp_expression(arg) for arg in mosp_data[operation]]
        first_arg, *remaining_args = parsed_arguments
        return MathematicalExpression(_MospOperationSql[operation], first_arg, remaining_args)
    elif operation in _MospBinaryOperations:
        return BooleanExpression(_parse_mosp_predicate(mosp_data))

    # parse aggregate (COUNT / AVG / MIN / ...) or function call (CURRENT_DATE() etc)
    arguments = mosp_data[operation] if mosp_data[operation] else []
    if isinstance(arguments, list):
        parsed_arguments = [_parse_mosp_expression(arg) for arg in arguments]
    else:
        parsed_arguments = [_parse_mosp_expression(arguments)]
    return FunctionExpression(operation, parsed_arguments, distinct=distinct)


def _parse_with_query(mosp_data: dict) -> WithQuery:
    """Parsing logic for individual ``WITH`` queries.

    Parameters
    ----------
    mosp_data : dict
        The subquery dictionary consisting of a ``"name"`` key that denotes the target name of the CTE and a
        ``"value"`` key that contains the actual query

    Returns
    -------
    WithQuery
        The parsed query
    """
    target_name = mosp_data["name"]
    mosp_query = mosp_data["value"]
    parsed_query = _MospQueryParser(mosp_query).parse_query()
    return WithQuery(parsed_query, target_name)


def _parse_cte_clause(mosp_data: dict | list) -> CommonTableExpression:
    """Parsing logic for an entire CTE.

    Parameters
    ----------
    mosp_data : dict | list
        The contents of the ``"with"`` key, i.e. the mapping has to be resolved already. For a single CTE, this will be
        a dictionary, for multiple CTEs a list.

    Returns
    -------
    CommonTableExpression
        The parsed CTE

    Raises
    ------
    ValueError
        If the `mosp_data` is neither a list nor a dictionary. We never encountered this case during testing. If it can
        be emitted for a valid query, we need to extend our parsing logic.
    """
    if isinstance(mosp_data, list):
        with_queries = [_parse_with_query(mosp_with) for mosp_with in mosp_data]
    elif isinstance(mosp_data, dict):
        with_queries = [_parse_with_query(mosp_data)]
    else:
        raise ValueError("Unknown WITH format: " + str(mosp_data))
    return CommonTableExpression(with_queries)


def _parse_window_function(mosp_data: dict) -> WindowExpression:
    """Parsing logic for window functions.

    Parameters
    ----------
    mosp_data : dict
        The mo-sql contents of the window function. This is a dictionary mapping a single key (the window function's
        operator) to the contents.

    Returns
    -------
    WindowFunctionExpression
        The parsed window function
    """
    function = _parse_mosp_expression(mosp_data["value"])
    mosp_window = mosp_data["over"]
    if "partitionby" in mosp_window:
        mosp_partition = mosp_window["partitionby"]
        partition_targets = ([_parse_mosp_expression(partition) for partition in mosp_partition]
                             if isinstance(mosp_partition, list) else [_parse_mosp_expression(mosp_partition)])
    else:
        partition_targets = []
    if "orderby" in mosp_window:
        orderby = _parse_orderby_clause(mosp_window["orderby"])
    else:
        orderby = None
    # Window function filters (e.g. SUM(salary) FILTER (WHERE salary > 100) OVER()) are currently not supported by mosp
    return WindowExpression(function, partitioning=partition_targets, ordering=orderby, filter_condition=None)


def _parse_case_expression(mosp_data: dict | list) -> CaseExpression:
    """Parsing logic for ``CASE`` expressions.

    Parameters
    ----------
    mosp_data : dict | list
        The mosql encoding of the case expression. This is the value of the ``"case"`` key.

    Returns
    -------
    CaseExpression
        The parsed case expression
    """
    if isinstance(mosp_data, dict):
        case_condition = _parse_mosp_predicate(mosp_data["when"])
        case_result = _parse_mosp_expression(mosp_data["then"])
        cases = [(case_condition, case_result)]
        return CaseExpression(cases)

    cases: list[tuple[AbstractPredicate, SqlExpression]] = []
    else_result: SqlExpression | None = None
    for mosp_case in mosp_data:
        if not isinstance(mosp_case, dict) or "when" not in mosp_case:
            else_result = _parse_mosp_expression(mosp_case)
            break
        case_condition = _parse_mosp_predicate(mosp_case["when"])
        case_result = _parse_mosp_expression(mosp_case["then"])
        cases.append((case_condition, case_result))
    return CaseExpression(cases, else_expr=else_result)


def _parse_select_statement(mosp_data: dict | str) -> BaseProjection:
    """Parsing logic for a single projection of the ``SELECT`` clause.

    The method basically tries to infer which kind of projection is described by the input data and potentially
    forwards the parsing to more specialized methods.

    Parameters
    ----------
    mosp_data : dict | str
        The complete projection data

    Returns
    -------
    BaseProjection
        The parsed projection.
    """
    if isinstance(mosp_data, dict) and "value" in mosp_data:
        # TODO: Why do we need to copy here? Leaving this in in case of legacy reasons or weird interactions.
        select_target = copy.copy(mosp_data["value"])
        target_name = mosp_data.get("name", None)
        parsed_target = _parse_mosp_expression(mosp_data) if "over" in mosp_data else _parse_mosp_expression(select_target)
        return BaseProjection(parsed_target, target_name)
    elif isinstance(mosp_data, dict) and "all_columns" in mosp_data:
        return BaseProjection.star()
    if mosp_data == "*":
        return BaseProjection.star()
    target_column = _parse_column_reference(mosp_data)
    return BaseProjection(ColumnExpression(target_column))


def _parse_select_clause(mosp_data: dict) -> Select:
    """Parsing logic for the ``SELECT`` clause.

    This determines the select type and delegates to specialized methods to handle the parsing of the individual
    projections.

    Parameters
    ----------
    mosp_data : dict
        The entire select data. This includes the main key to access the ``SELECT`` clause since we need to determine
        which kind of selection we should perform. This information is encoded in the key.

    Returns
    -------
    Select
        The parsed clause

    Raises
    ------
    ValueError
        If the selection type is unknown. We never encountered this case during testing. If it can be emitted for a
        valid query, we need to extend our parsing logic.
    ValueError
        If the structure of the projection is unknown. Currently we recognize lists, dictionaries and strings. We
        never encountered anything else during testing. If it can be emitted for a valid query, we need to extend our
        parsing logic.
    """
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

    return Select(parsed_targets, _MospSelectTypes[select_type])


_TableReferencePattern = re.compile(r"(?P<full_name>\S+)( (AS )?(?P<alias>\S+))?")
"""Regular expression to parse table names and aliases.

This pattern is necessary because mo-sql appears to be inconsistent in the way tables with aliases are encoded. During
testing (at least in older versions of mo-sql) we encountered cases where the table would be split into a dictionary
with keys for the actual table name and the alias, but also cases where name and alias where put into a single string.

To be better safe than sorry we try to handle both cases by using the regular expression.

References
----------

.. Pattern debugging: https://regex101.com/r/HdKzQg/2
"""


def _parse_table_reference(table: str | dict, *,
                           cte_tables: Optional[dict[str, TableReference]] = None) -> TableReference:
    """Parsing logic to generate table references

    Parameters
    ----------
    table : str | dict
        The table encoding

    Returns
    -------
    TableReference
        The parsed table

    Raises
    ------
    KeyError
        If the table is encoded as a dictionary, but does not contain the required keys (i.e. the ``"value"`` key). We
        never encountered this case during testing. If it can be emitted for a valid query, we need to extend our
        parsing logic.
    ValueError
        If the table is not a dictionary, nor does it match the `_TableReferencePattern`. We never encountered this
        case during testing. If it can be emitted for a valid query, we need to extend our parsing logic.
    """
    # mo_sql table format
    if isinstance(table, dict):
        table_name = table["value"]
        table_alias = table.get("name", None)
        if cte_tables and table_name in cte_tables:
            return TableReference(cte_tables[table_name], table_alias, True)
        return TableReference(table_name, table_alias)

    # string-based table format
    pattern_match = _TableReferencePattern.match(table)
    if not pattern_match:
        raise ValueError(f"Could not parse table reference for '{table}'")
    full_name, alias = pattern_match.group("full_name", "alias")
    alias = "" if not alias else alias
    table = TableReference(full_name, alias)
    if cte_tables and full_name in cte_tables:
        return TableReference(full_name, alias, True)
    return table


def _parse_column_reference(column: str) -> ColumnReference:
    """Parsing logic to generate column references.

    This tries to setup a column with its associated table if the encoding contains a dot (``.``). Otherwise, it
    assumes that the entire encoding is only the column name.

    Parameters
    ----------
    column : str
        The column encoding

    Returns
    -------
    ColumnReference
        The parsed column reference

    Notes
    -----
    The first case is the reason for why we need the subsequent binding process when parsing a query. When creating the
    column reference, we don't know anything about the context of the query (due to the design of our parser as a
    nesting of function calls that do not share any state). When we encouter a column that also contains a table, this
    is usually the tables alias. But in the parsing function we don't know about the tables full name. Hence, we can
    only set the alias part of the table reference. However, based on the ``FROM`` clause, both components of the table
    reference can be inferred. At this point, we have two table references that encode to the same table on a logical
    level, but not on a code level since they contain different information (full name and alias vs. just the alias).
    The binding process is there to reconcile this situation and normalize all table references.
    """
    if "." not in column:
        return ColumnReference(column)

    table, column = column.split(".")
    table_ref = TableReference("", table)
    return ColumnReference(column, table_ref)


def _parse_implicit_from_clause(mosp_data: dict, *, cte_tables: Optional[dict[str, TableReference]] = None
                                ) -> Optional[ImplicitFromClause]:
    """Parsing logic for ``FROM`` clauses that only reference plain tables and do not contain subqueries.

    Parameters
    ----------
    mosp_data : dict
        The entire ``FROM`` clause, i.e. starting at and including the ``"from"`` key.
    cte_tables : Optional[dict[str, TableReference]], optional
        Tables that have been exported by a *WITH* statement. These can also appear in the *FROM* clause and should be treated
        as virtual tables. The default assumption is that there are no such tables.

    Returns
    -------
    Optional[ImplicitFromClause]
        The parsed clause. This can be ``None`` if the `mosp_data` does not contain a ``"from"`` key.

    Raises
    ------
    ValueError
        If the `mosp_data` does not map the ``"from"`` key to a string, a dictionary, or a map. We never encountered
        this case during testing. If it can be emitted for a valid query, we need to extend our parsing logic.

    See Also
    --------
    _is_implicit_query
    """
    if "from" not in mosp_data:
        return None
    from_clause = mosp_data["from"]
    if isinstance(from_clause, (str, dict)):
        table = _parse_table_reference(from_clause, cte_tables=cte_tables)
        return ImplicitFromClause(DirectTableSource(table))
    elif not isinstance(from_clause, list):
        raise ValueError("Unknown FROM clause structure: " + str(from_clause))
    parsed_sources = [DirectTableSource(_parse_table_reference(table, cte_tables=cte_tables))
                      for table in from_clause]
    return ImplicitFromClause(parsed_sources)


def _parse_explicit_from_clause(mosp_data: dict, *,
                                cte_tables: Optional[dict[str, TableReference]] = None) -> ExplicitFromClause:
    """Parsing logic for ``FROM`` clauses that exclusively make use of the ``JOIN ON`` syntax.

    In contrast to implicit ``FROM`` clauses, their explicit counterparts can join both plain tables as well as
    subqueries.

    Parameters
    ----------
    mosp_data : dict
        The entire ``FROM`` clause, i.e. starting at and including the ``"from"`` key.
    cte_tables : Optional[dict[str, TableReference]], optional
        Tables that have been exported by a *WITH* statement. These can also appear in the *FROM* clause and should be treated
        as virtual tables. The default assumption is that there are no such tables.

    Returns
    -------
    ExplicitFromClause
        The parsed clause

    Raises
    ------
    ValueError
        If there is no ``"from"`` key in the `mosp_data`
    ValueError
        If the ``"from"`` key does not map to a list of table items.
    ValueError
        If the joined table is neither encoded by a list, a dictionary, or a string. We never encountered this case
        during testing. If it can be emitted for a valid query, we need to extend our parsing logic.

    See Also
    --------
    _is_explicit_query
    """
    if "from" not in mosp_data:
        raise ValueError("No tables in FROM clause")
    from_clause = mosp_data["from"]
    if not isinstance(from_clause, list):
        raise ValueError("Unknown FROM clause format: " + str(from_clause))
    first_table, *joined_tables = from_clause
    initial_table = _parse_table_reference(first_table, cte_tables=cte_tables)
    parsed_joins = [_parse_explicit_join(joined_table, cte_tables=cte_tables) for joined_table in joined_tables]
    return ExplicitFromClause(DirectTableSource(initial_table), parsed_joins)


def _parse_explicit_join(joined_table: dict, *,
                         cte_tables: Optional[dict[str, TableReference]] = None) -> JoinTableSource:
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
        return JoinTableSource(SubqueryTableSource(joined_subquery, join_alias), join_condition,
                               join_type=parsed_join_type)
    elif isinstance(join_source, dict):
        # we found a normal table join with an alias
        parsed_table = _parse_table_reference(join_source, cte_tables=cte_tables)
        return JoinTableSource(DirectTableSource(parsed_table), join_condition, join_type=parsed_join_type)
    elif isinstance(join_source, str):
        # we found a normal table join without an alias
        parsed_table = _parse_table_reference(join_source, cte_tables=cte_tables)
        return JoinTableSource(DirectTableSource(parsed_table), join_condition, join_type=parsed_join_type)
    elif isinstance(join_source, list):
        # we found an join with a nested JOIN statement
        base_table, *nested_joins = join_source
        parsed_table = DirectTableSource(_parse_table_reference(base_table, cte_tables=cte_tables))
        nested_tables = [_parse_explicit_join(nested_join, cte_tables=cte_tables) for nested_join in nested_joins]
        return JoinTableSource(parsed_table, join_condition, joined_tables=nested_tables, join_type=parsed_join_type)
    else:
        raise ValueError("Unknown JOIN format: " + str(joined_table))


def _parse_base_table_source(mosp_data: dict | str, *,
                             cte_tables: Optional[dict[str, TableReference]] = None
                             ) -> DirectTableSource | SubqueryTableSource:
    """Parsing logic for table sources that do not use the ``JOIN ON`` syntax.

    This method will determine the appropriate type of source (pure table or subquery) and potentially delegate its
    construction to a dedicated method.

    Parameters
    ----------
    mosp_data : dict | str
        The table encoding. Strings are direct sources, whereas dicts can encode both direct table sources as well as
        subqueries.
    cte_tables : Optional[dict[str, TableReference]], optional
        Tables that have been exported by a *WITH* statement. These can also appear in the *FROM* clause and should be treated
        as virtual tables. The default assumption is that there are no such tables.

    Returns
    -------
    DirectTableSource | SubqueryTableSource
        The parsed source

    Raises
    ------
    TypeError
        If the `mosp_data` is neither a string nor a dict, or the dict does not contain the required keys. These keys
        are ``"name"`` and ``"value"``. We never encountered this case during testing. If it can be emitted for a valid
        query, we need to extend our parsing logic.
    TypeError
        If the `mosp_data` is a dictionary, but its ``"value"`` key is neither a string (indicating a direct table
        source), nor a valid subquery (i.e. a dictionary with a valid ``SELECT`` key). We never encountered this case
        during testing. If it can be emitted for a valid query, we need to extend our parsing logic.
    """
    if isinstance(mosp_data, str):
        return DirectTableSource(_parse_table_reference(mosp_data, cte_tables=cte_tables))
    if not isinstance(mosp_data, dict) or "value" not in mosp_data or "name" not in mosp_data:
        raise TypeError("Unknown FROM clause target: " + str(mosp_data))

    value_target = mosp_data["value"]
    if isinstance(value_target, str):
        return DirectTableSource(_parse_table_reference(mosp_data, cte_tables=cte_tables))
    is_subquery_table = (isinstance(value_target, dict)
                         and any(select_type in value_target for select_type in _MospSelectTypes))
    if not is_subquery_table:
        raise TypeError("Unknown FROM clause target: " + str(mosp_data))
    parsed_subquery = _MospQueryParser(value_target, mosp.format(value_target)).parse_query()
    subquery_target = mosp_data["name"]
    return SubqueryTableSource(parsed_subquery, subquery_target)


def _parse_join_table_source(mosp_data: dict, *,
                             cte_tables: Optional[dict[str, TableReference]] = None) -> JoinTableSource:
    """Parsing logic for a single ``JOIN ON``statement in the ``FROM`` clause.

    This method will determine the precise kind of join being used (which are encoded as part of the key in mo-sql) and
    delegate to dedicated procedures to handle the parsing of the referenced table (subquery or direct table) and the
    join predicate (if there is one).

    Parameters
    ----------
    mosp_data : dict
        The join encoding. This dictionary has to contain the key describing the precise join type at the root level.
    cte_tables : Optional[dict[str, TableReference]], optional
        Tables that have been exported by a *WITH* statement. These can also appear in the *FROM* clause and should be treated
        as virtual tables. The default assumption is that there are no such tables.

    Returns
    -------
    JoinTableSource
        The parsed table source

    See Also
    --------
    _MospJoinTypes
    """
    join_type = next(jt for jt in _MospJoinTypes.keys() if jt in mosp_data)
    join_target = mosp_data[join_type]
    parsed_target = _parse_base_table_source(join_target)
    parsed_join_type = _MospJoinTypes[join_type]
    join_condition = _parse_mosp_predicate(mosp_data["on"]) if "on" in mosp_data else None
    return JoinTableSource(parsed_target, join_condition, join_type=parsed_join_type)


def _parsed_mixed_from_clause(mosp_data: dict, *, cte_tables: Optional[dict[str, TableReference]] = None) -> From:
    """Parsing logic for ``FROM`` clauses that consist of pure table sources, explicit ``JOIN``s and subqueries.

    The method will figure out the most appropriate representation for each source and delegate its construction to a
    dedicated method.

    Parameters
    ----------
    mosp_data : dict
        The entire ``FROM`` clause, i.e. starting at and including the ``"from"`` key.
    cte_tables : Optional[dict[str, TableReference]], optional
        Tables that have been exported by a *WITH* statement. These can also appear in the *FROM* clause and should be treated
        as virtual tables. The default assumption is that there are no such tables.

    Returns
    -------
    From
        The parsed clause

    Raises
    ------
    ValueError
        If the the clause available under the ``"from"`` key is of unknown structure. The parser expects it to either
        be a string, a dictionary, or a list. We never encountered this case during testing. If it can be emitted for a
        valid query, we need to extend our parsing logic.
    """
    if "from" not in mosp_data:
        return From([])
    from_clause = mosp_data["from"]

    if isinstance(from_clause, str):
        return From(DirectTableSource(_parse_table_reference(from_clause, cte_tables=cte_tables)))
    elif isinstance(from_clause, dict):
        return From(_parse_base_table_source(from_clause, cte_tables=cte_tables))
    elif not isinstance(from_clause, list):
        raise ValueError("Unknown FROM clause type: " + str(from_clause))

    parsed_from_clause_entries = []
    for entry in from_clause:
        join_entry = isinstance(entry, dict) and any(join_type in entry for join_type in _MospJoinTypes)
        parsed_entry = (_parse_join_table_source(entry, cte_tables=cte_tables) if join_entry
                        else _parse_base_table_source(entry, cte_tables=cte_tables))
        parsed_from_clause_entries.append(parsed_entry)
    return From(parsed_from_clause_entries)


def _parse_groupby_clause(mosp_data: dict | list) -> GroupBy:
    """Parsing logic for ``GROUP BY``

    Parameters
    ----------
    mosp_data : dict | list
        The enconded value under the ``GROUP BY`` key.

    Returns
    -------
    GroupBy
        The parsed clause
    """

    # The format of GROUP BY clauses is a bit weird in mo-sql. Therefore, the parsing logic looks quite hacky
    # Take a look at the MoSQLParsingTests for details.

    if isinstance(mosp_data, list):
        columns = [_parse_mosp_expression(col["value"]) for col in mosp_data]
        distinct = False
        return GroupBy(columns, distinct)

    mosp_data = mosp_data["value"]
    if "distinct" in mosp_data:
        groupby_clause = _parse_groupby_clause(mosp_data["distinct"])
        groupby_clause = GroupBy(groupby_clause.group_columns, True)
        return groupby_clause
    else:
        columns = [_parse_mosp_expression(mosp_data)]
        return GroupBy(columns, False)


def _parse_having_clause(mosp_data: dict) -> Having:
    """Parsing logic for ``HAVING`` clauses.

    Parameters
    ----------
    mosp_data : dict
        The encoded value under the ``HAVING`` key. This is expected to be a regular encoded predicate.

    Returns
    -------
    Having
        The parsed clause
    """
    return Having(_parse_mosp_predicate(mosp_data))


def _parse_orderby_expression(mosp_data: dict) -> OrderByExpression:
    """Parsing logic for individual parts of ``ORDER BY`` clauses.

    Parameters
    ----------
    mosp_data : dict
        The encoded ordering expressions

    Returns
    -------
    OrderByExpression
        The parsed expression
    """
    column = _parse_mosp_expression(mosp_data["value"])
    ascending = mosp_data["sort"] == "asc" if "sort" in mosp_data else None
    return OrderByExpression(column, ascending)


def _parse_orderby_clause(mosp_data: dict | list) -> OrderBy:
    """Parsing logic for entire ``ORDER BY`` clauses

    Parameters
    ----------
    mosp_data : dict | list
        The encoded value under the ``ORDER BY`` key.

    Returns
    -------
    OrderBy
        The parsed clause
    """
    if isinstance(mosp_data, list):
        order_expressions = [_parse_orderby_expression(order_expr) for order_expr in mosp_data]
    else:
        order_expressions = [_parse_orderby_expression(mosp_data)]
    return OrderBy(order_expressions)


def _parse_limit_clause(mosp_data: dict) -> Limit:
    """Parsing logic for ``LIMIT`` clauses.

    Parameters
    ----------
    mosp_data : dict
        The encoding of the entire query. This is necessary because the `Limit` clause combines two separate clauses:
        the actual limit, as well as the offset.

    Returns
    -------
    Limit
        The parsed clause.

    Raises
    ------
    ValueError
        If the query encoding contains neither a ``"limit"`` key, nor an ``"offset"`` key.
    """

    # TODO: add support for FETCH FIRST syntax

    limit = mosp_data.get("limit", None)
    offset = mosp_data.get("offset", None)
    return Limit(limit=limit, offset=offset)


def _is_implicit_query(mosp_data: dict) -> bool:
    """Checks, if an encoded query defines an `ImplicitFromClause`.

    Parameters
    ----------
    mosp_data : dict
        The encoded query to check

    Returns
    -------
    bool
        Whether the query contains a valid implicit from clause. As a special case, a query without a ``FROM`` clause
        is considered a valid implicit clause.
    """
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
    """Checks, if an encoded query defines an `ExplicitFromClause`.

    Queries without a ``FROM`` clause fail this test.

    Parameters
    ----------
    mosp_data : dict
        The encoded query to check

    Returns
    -------
    bool
        Whether the query contains a valid explicit from clause.
    """
    if "from" not in mosp_data:
        return False

    from_clause = mosp_data["from"]
    if not isinstance(from_clause, list):
        return False

    join_statements = from_clause[1:]
    for table in join_statements:
        if isinstance(table, str):
            return False
        explicit_join = isinstance(table, dict) and any(join_type in table for join_type in _MospJoinTypes)
        subquery_table = (isinstance(table, dict)
                          and (any(select_type in table for select_type in _MospSelectTypes) or "value" in table))
        if not explicit_join or subquery_table:
            return False
    return True


class _MospQueryParser:
    """The parser class acts as a one-stop-shop to parse input queries.

    It handles all required preparatory steps and manages the necessary state to perform the parsing process. Notice
    that this state does not correspond to state that is carried through the different parsing methods. Instead, this
    means additional parts of the query that have to be maintained separately due to mo-sql oddities.

    Parameters
    ----------
    mosp_data : dict
        The query, as encoded by the mo-sql parsing functionality.
    raw_query : str, optional
        The original query. This is just intended for debugging to cross-reference weird mosp-output to the actual
        query. Defaults to an empty string.
    """

    def __init__(self, mosp_data: dict, raw_query: str = "") -> None:
        self._raw_query = raw_query
        self._mosp_data = mosp_data
        self._explain = {}

        self._prepare_query()

    def parse_query(self) -> SqlQuery:
        """Handles the parsing process for the entire query.

        Returns
        -------
        SqlQuery
            The parsed query
        """
        if "with" in self._mosp_data:
            cte_clause = _parse_cte_clause(self._mosp_data["with"])
            cte_exported_tables = {cte.target_name: cte.target_table for cte in cte_clause.queries}
        else:
            cte_clause = None
            cte_exported_tables = {}

        if _is_implicit_query(self._mosp_data):
            implicit, explicit = True, False
            from_clause = _parse_implicit_from_clause(self._mosp_data, cte_tables=cte_exported_tables)
        elif _is_explicit_query(self._mosp_data):
            implicit, explicit = False, True
            from_clause = _parse_explicit_from_clause(self._mosp_data, cte_tables=cte_exported_tables)
        else:
            implicit, explicit = False, False
            from_clause = _parsed_mixed_from_clause(self._mosp_data, cte_tables=cte_exported_tables)
        select_clause = _parse_select_clause(self._mosp_data)
        where_clause = _parse_where_clause(self._mosp_data["where"]) if "where" in self._mosp_data else None

        # TODO: support for EXPLAIN queries

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
            return ImplicitSqlQuery(select_clause=select_clause, from_clause=from_clause,
                                    where_clause=where_clause,
                                    groupby_clause=groupby_clause, having_clause=having_clause,
                                    orderby_clause=orderby_clause, limit_clause=limit_clause,
                                    cte_clause=cte_clause)
        elif not implicit and explicit:
            return ExplicitSqlQuery(select_clause=select_clause, from_clause=from_clause,
                                    where_clause=where_clause,
                                    groupby_clause=groupby_clause, having_clause=having_clause,
                                    orderby_clause=orderby_clause, limit_clause=limit_clause,
                                    cte_clause=cte_clause)
        else:
            return MixedSqlQuery(select_clause=select_clause, from_clause=from_clause,
                                 where_clause=where_clause,
                                 groupby_clause=groupby_clause, having_clause=having_clause,
                                 orderby_clause=orderby_clause, limit_clause=limit_clause,
                                 cte_clause=cte_clause)

    def _prepare_query(self) -> None:
        """Performs necessary pre-processing steps before the actual parsing can take place.

        Currently, this pre-processing is only focussed on dealing with ``EXPLAIN`` queries, since the ``EXPLAIN``
        key is used as another indirection layer before the actual query encoding can be accessed. This method takes
        care of dismantling that indirection and setting the `_mosp_data` and `_explain` attributes accordingly.
        Afterwards, the `_mosp_data` contains the encoded query at the top level.
        """
        if "explain" in self._mosp_data:
            self._explain = {"analyze": self._mosp_data.get("analyze", False),
                             "format": self._mosp_data.get("format", "text")}
            self._mosp_data = self._mosp_data["explain"]


def bind_column_references(query: QueryType, *, with_schema: bool = True,
                           db_schema: Optional["DatabaseSchema"] = None) -> QueryType:  # noqa: F821 # type: ignore
    """Determines the tables that each column belongs to and sets the appropriate references.

    This binding of columns to their tables happens in two phases: During the first phase, a *syntactic* binding is performed.
    This operates on column names of the form ``<alias>.<column name>``, where ``<alias>`` is either an actual alias of a table
    from the ``FROM`` clause, or the full name of such a table. For all such names, the reference is set up directly.
    During the second phase, a *schema* binding is performed. This is applied to all columns that could not be bound during the
    first phase and involves querying the schema catalog of a live database. It determines which of the tables from the
    ``FROM`` clause contain a column with a name similar to the name of the unbound column and sets up the corresponding table
    reference. If multiple tables contain a specific column, any of them might be chosen. The second phase is entirely
    optional and can be skipped altogether. In this case, some columns might end up without a valid table reference, however.
    This in turn might break some applications.

    Parameters
    ----------
    query : QueryType
        The query whose columns should be bound
    with_schema : bool, optional
        Whether the second binding phase based on the schema catalog of a live database should be performed. This is enabled by
        default
    db_schema : Optional[DatabaseSchema], optional
        The schema to use for the second binding phase. If `with_schema` is enabled, but this parameter is ``None``, the schema
        is inferred based on the current database of the `DatabasePool`. This defaults to ``None``.

    Returns
    -------
    QueryType
        The updated query. Notice that some columns might still remain unbound if none of the phases was able to find a table.
    """
    from ..db import DatabasePool

    if not query.from_clause:
        return query

    table_alias_map: dict[str, TableReference] = {}
    unbound_tables: set[TableReference] = set()
    pure_virtual_tables: set[TableReference] = set()
    if query.cte_clause:
        for cte in query.cte_clause.queries:
            pure_virtual_tables.add(cte.target_table)
    for table_source in query.from_clause.items:
        if isinstance(table_source, SubqueryTableSource):
            pure_virtual_tables.add(table_source.target_table)

    for table in query.tables():
        if table in pure_virtual_tables:
            table_alias_map[table.alias] = table

        if table.full_name and table.alias:
            table_alias_map[table.full_name] = table
            table_alias_map[table.alias] = table
        elif table.full_name:
            table_alias_map[table.full_name] = table
        else:
            unbound_tables.add(table)
    for table in unbound_tables:
        if table.alias not in table_alias_map:
            table_alias_map[table.alias] = table

    unbound_columns: list[ColumnReference] = []
    necessary_renamings: dict[ColumnReference, ColumnReference] = {}
    for column in query.columns():
        if not column.table:
            unbound_columns.append(column)
        elif column.table.identifier() in table_alias_map:
            bound_column = ColumnReference(column.name, table_alias_map[column.table.identifier()])
            necessary_renamings[column] = bound_column

    partially_bound_query = transform.rename_columns_in_query(query, necessary_renamings)
    if not with_schema:
        return partially_bound_query

    db_schema = db_schema if db_schema else DatabasePool().get_instance().current_database().schema()
    candidate_tables = [table for table in query.tables() if table.full_name]
    unbound_renamings: dict[ColumnReference, ColumnReference] = {}
    for column in unbound_columns:
        try:
            target_table = db_schema.lookup_column(column, candidate_tables)
            bound_column = ColumnReference(column.name, target_table)
            unbound_renamings[column] = bound_column
        except ValueError:
            # A ValueError is raised if the column is not found in any of the tables. However, this can still be
            # a valid query, e.g. a dependent subquery. Therefore, we simply ignore this error and leave the column
            # unbound.
            pass
    return transform.rename_columns_in_query(partially_bound_query, unbound_renamings)


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

    from ..db import DatabasePool  # local import to prevent circular imports

    bind_columns = bind_columns if bind_columns is not None else auto_bind_columns
    db_schema = (db_schema if db_schema or not bind_columns
                 else DatabasePool.get_instance().current_database().schema())
    mosp_data = mosp.parse(query)
    parsed_query = _MospQueryParser(mosp_data, query).parse_query()
    if _skip_all_binding:
        return parsed_query
    return bind_column_references(parsed_query, with_schema=bind_columns, db_schema=db_schema)


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
