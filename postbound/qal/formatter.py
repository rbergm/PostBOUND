"""Provides logic to generate pretty strings for query objects."""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Optional

from . import transform
from ._qal import (
    SqlExpression, SubqueryExpression, CaseExpression, Limit, CommonTableExpression,
    CompoundOperator,
    ValuesWithQuery,
    From, ImplicitFromClause, ExplicitFromClause,
    TableSource, DirectTableSource, JoinTableSource, SubqueryTableSource, ValuesTableSource,
    Select, Hint, Where, GroupBy, UnionClause, IntersectClause, ExceptClause,
    SelectType,
    AbstractPredicate, CompoundPredicate,
    SqlQuery, SetQuery, SelectStatement,
    quote
)
from ..util.errors import InvariantViolationError

FormatIndentDepth = 2
"""The default amount of whitespace that is used to indent specific parts of the SQL query."""


def _increase_indentation(content: str, indentation: int = FormatIndentDepth) -> str:
    """Prefixes all lines in a string by a given amount of whitespace.

    This breaks the input string into separate lines and adds whitespace equal to the desired amount at the start of
    each line.

    Parameters
    ----------
    content : str
        The string that should be prefixed/whose indentation should be increased.
    indentation : Optional[int], optional
        The amount of whitespace that should be added, by default `FormatIndentDepth`

    Returns
    -------
    str
        The indented string
    """
    indent_prefix = "\n" + " " * indentation
    return " " * indentation + indent_prefix.join(content.split("\n"))


class FormattingSubqueryExpression(SubqueryExpression):
    """Wraps subquery expressions to ensure that they are also pretty-printed and aligned properly.

    This class acts as a decorator around the actual subquery. It can be used entirely as a replacement of the original
    query.

    Parameters
    ----------
    original_expression : SubqueryExpression
        The actual subquery.
    inline_hint_block : bool
        Whether potential hint blocks of the subquery should be printed as preceding blocks or as inline blocks (see
        `format_quick` for details)
    indentation : int
        The current amount of indentation that should be used for the subquery. While pretty-printing, additional
        indentation levels can be inserted for specific parts of the query.
    """
    def __init__(self, original_expression: SubqueryExpression, inline_hint_block: bool, indentation: int) -> None:
        super().__init__(original_expression.query)
        self._inline_hint_block = inline_hint_block
        self._indentation = indentation

    def __str__(self) -> str:
        formatted = "(" + format_quick(self.query, inline_hint_block=self._inline_hint_block, trailing_semicolon=False) + ")"
        prefix = " " * (self._indentation + 2)
        if "\n" not in formatted:
            return prefix + formatted

        lines = formatted.split("\n")
        indented_lines = [f"\n{prefix}{lines[0]}"] + [prefix + line for line in lines[1:]]
        return "\n".join(indented_lines)


class FormattingCaseExpression(CaseExpression):
    def __init__(self, original_expression: CaseExpression, indentation: int) -> None:
        super().__init__(original_expression.cases, else_expr=original_expression.else_expression)
        self._indentation = indentation

    def __str__(self) -> str:
        case_indentation = " " * (self._indentation + 2)
        case_block_entries: list[str] = ["CASE"]
        for case, value in self.cases:
            case_block_entries.append(f"{case_indentation}WHEN {case} THEN {value}")
        if self.else_expression is not None:
            case_block_entries.append(f"{case_indentation}ELSE {self.else_expression}")
        case_block_entries.append("END")
        return "\n".join(case_block_entries)


class FormattingLimitClause(Limit):
    """Wraps the `Limit` clause to enable pretty printing of its different parts (limit and offset).

    This class acts as a decorator around the actual clause. It can be used entirely as a replacement of the original
    clause.

    Parameters
    ----------
    original_clause : clauses.Limit
        The clause to wrap
    """

    def __init__(self, original_clause: Limit) -> None:
        super().__init__(limit=original_clause.limit, offset=original_clause.offset)

    def __str__(self) -> str:
        if self.offset and self.limit:
            return f"OFFSET {self.offset} ROWS\nFETCH FIRST {self.limit} ROWS ONLY"
        elif self.offset:
            return f"OFFSET {self.offset} ROWS"
        elif self.limit:
            return f"FETCH FIRST {self.limit} ROWS ONLY"
        raise InvariantViolationError("Either limit or offset must be specified for Limit clause")


def _quick_format_cte(cte_clause: CommonTableExpression) -> list[str]:
    """Formatting logic for Common Table Expressions

    Parameters
    ----------
    cte_clause : clauses.CommonTableExpression
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the CTE, indented as necessary.
    """
    recursive_info = "RECURSIVE " if cte_clause.recursive else ""

    if len(cte_clause.queries) == 1:
        cte_query = cte_clause.queries[0]
        if isinstance(cte_query, ValuesWithQuery):
            cte_header = "WITH "
            cte_content = str(cte_query)
        else:
            mat_info = "" if cte_query.materialized is None else ("MATERIALIZED " if cte_query.materialized
                                                                  else "NOT MATERIALIZED ")
            cte_header = f"WITH {recursive_info}{quote(cte_query.target_name)} AS {mat_info}("
            cte_content = format_quick(cte_query.query, trailing_semicolon=False)
        cte_content = _increase_indentation(cte_content)
        cte_footer = ")"
        return [cte_header, cte_content, cte_footer]

    first_cte, *remaining_ctes = cte_clause.queries
    first_content = _increase_indentation(format_quick(first_cte.query, trailing_semicolon=False))
    mat_info = "" if first_cte.materialized is None else ("MATERIALIZED " if first_cte.materialized else "NOT MATERIALIZED ")
    formatted_parts: list[str] = [f"WITH{recursive_info} {quote(first_cte.target_name)} AS {mat_info}(", first_content]
    for next_cte in remaining_ctes:
        mat_info = "" if next_cte.materialized is None else ("MATERIALIZED " if first_cte.materialized
                                                             else "NOT MATERIALIZED ")
        current_header = f"), {quote(next_cte.target_name)} AS {mat_info}("
        cte_content = _increase_indentation(format_quick(next_cte.query, trailing_semicolon=False))

        formatted_parts.append(current_header)
        formatted_parts.append(cte_content)

    formatted_parts.append(")")
    return formatted_parts


def _quick_format_select(select_clause: Select, *,
                         inlined_hint_block: Optional[Hint] = None) -> list[str]:
    """Quick and dirty formatting logic for *SELECT* clauses.

    Up to 3 targets are put on the same line, otherwise each target is put on a separate line.

    Parameters
    ----------
    select_clause : Select
        The clause to format
    inlined_hint_block : Optional[Hint], optional
        A hint block that should be inserted after the *SELECT* statement. Defaults to *None* which indicates that
        no block should be inserted that way

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    hint_text = f"{inlined_hint_block} " if inlined_hint_block else ""
    if len(select_clause.targets) > 3:
        first_target, *remaining_targets = select_clause.targets
        formatted_targets = [f"SELECT {hint_text}{first_target}"
                             if select_clause.projection_type == SelectType.Select
                             else f"SELECT DISTINCT {hint_text}{first_target}"]
        formatted_targets += [((" " * FormatIndentDepth) + str(target)) for target in remaining_targets]
        for i in range(len(formatted_targets) - 1):
            formatted_targets[i] += ","
        return formatted_targets
    else:
        distinct_text = "DISTINCT " if select_clause.projection_type == SelectType.SelectDistinct else ""
        targets_text = ", ".join(str(target) for target in select_clause.targets)
        return [f"SELECT {distinct_text}{hint_text}{targets_text}"]


def _quick_format_implicit_from(from_clause: ImplicitFromClause) -> list[str]:
    """Quick and dirty formatting logic for implicit *FROM* clauses.

    Up to 3 tables are put on the same line, otherwise each table is put on its own line.

    Parameters
    ----------
    from_clause : ImplicitFromClause
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    tables = list(from_clause.itertables())
    if not tables:
        return []
    elif len(tables) > 3:
        first_table, *remaining_tables = tables
        formatted_tables = [f"FROM {first_table}"]
        formatted_tables += [((" " * FormatIndentDepth) + str(tab)) for tab in remaining_tables]
        for i in range(len(formatted_tables) - 1):
            formatted_tables[i] += ","
        return formatted_tables
    else:
        tables_str = ", ".join(str(tab) for tab in tables)
        return [f"FROM {tables_str}"]


def _quick_format_tablesource(table_source: TableSource) -> list[str]:
    """Quick and dirty formatting logic for table sources.

    Parameters
    ----------
    table_source : TableSource
        The table source to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the table source, indented as necessary.
    """

    prefix = " " * FormatIndentDepth
    match table_source:
        case DirectTableSource() | ValuesTableSource():
            return [str(table_source)]

        case SubqueryTableSource():
            elems: list[str] = ["LATERAL ("] if table_source.lateral else ["("]
            subquery_elems = format_quick(table_source.query, trailing_semicolon=False).split("\n")
            subquery_elems = [((" " * FormatIndentDepth) + str(child)) for child in subquery_elems]
            elems.extend(subquery_elems)
            elems.append(")")
            if table_source.target_name:
                elems[-1] += f" AS {quote(table_source.target_name)}"
            return elems

        case JoinTableSource():
            if isinstance(table_source.left, DirectTableSource) and isinstance(table_source.right, DirectTableSource):
                # case R JOIN S ON ...
                elems = [str(table_source.left), f"{prefix}{table_source.join_type} {table_source.right}"]
                if table_source.join_condition:
                    elems[-1] += f" ON {table_source.join_condition}"
                return elems

            if isinstance(table_source.left, JoinTableSource) and isinstance(table_source.right, DirectTableSource):
                # case R JOIN S ON ... JOIN T ON ...
                elems = _quick_format_tablesource(table_source.left)
                join_condition = f" ON {table_source.join_condition}" if table_source.join_condition else ""
                elems.append(f"{prefix}{table_source.join_type} {table_source.right}{join_condition}")
                return elems

            if isinstance(table_source.left, DirectTableSource) and isinstance(table_source.right, JoinTableSource):
                elems = [str(table_source.left)]
                right_children = _quick_format_tablesource(table_source.right)
                right_children[0] = f"{table_source.join_type} ({right_children[0]}"
                right_children[1:] = [((" " * FormatIndentDepth) + str(child)) for child in right_children[1:]]
                elems += right_children
                elems.append(")")
                if table_source.join_condition:
                    elems[-1] += f" ON {table_source.join_condition}"
                return elems

            elems: list[str] = []
            elems += _quick_format_tablesource(table_source.left)
            elems.append(f"{table_source.join_type}")
            elems += _quick_format_tablesource(table_source.right)
            if table_source.join_condition:
                elems[-1] += f" ON {table_source.join_condition}"
            elems = [((" " * FormatIndentDepth) + str(child)) for child in elems]
            return elems

        case _:
            raise ValueError("Unsupported table source type: " + str(table_source))


def _quick_format_explicit_from(from_clause: ExplicitFromClause) -> list[str]:
    """Quick and dirty formatting logic for explicit *FROM* clauses.

    This function just puts each *JOIN ON* statement on a separate line.

    Parameters
    ----------
    from_clause : ExplicitFromClause
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    items = _quick_format_tablesource(from_clause.root)
    items[0] = f"FROM {items[0]}"
    return items


def _quick_format_general_from(from_clause: From) -> list[str]:
    """Quick and dirty formatting logic for general *FROM* clauses.

    This function just puts each part of the *FROM* clause on a separate line.

    Parameters
    ----------
    from_clause : From
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    elems: list[str] = ["FROM"]
    for table_source in from_clause.items:
        current_elems = _quick_format_tablesource(table_source)
        current_elems = [((" " * FormatIndentDepth) + str(child)) for child in current_elems]
        current_elems[-1] += ","
        elems += current_elems
    elems[-1] = elems[-1].removesuffix(",")
    return elems


def _quick_format_predicate(predicate: AbstractPredicate) -> list[str]:
    """Quick and dirty formatting logic for arbitrary (i.e. also compound) predicates.

    *AND* conditions are put on separate lines, everything else is put on one line.

    Parameters
    ----------
    predicate : AbstractPredicate
        The predicate to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the predicate, indented as necessary.
    """
    if not isinstance(predicate, CompoundPredicate):
        return [str(predicate)]
    compound_pred: CompoundPredicate = predicate
    if compound_pred.operation == CompoundOperator.And:
        first_child, *remaining_children = compound_pred.children
        return [str(first_child)] + ["AND " + str(child) for child in remaining_children]
    return [str(compound_pred)]


def _quick_format_where(where_clause: Where) -> list[str]:
    """Quick and dirty formatting logic for *WHERE* clauses.

    This function just puts each part of an *AND* condition on a separate line and leaves the parts of *OR*
    conditions, negations or base predicates on the same line.

    Parameters
    ----------
    where_clause : Where
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    first_pred, *additional_preds = _quick_format_predicate(where_clause.predicate)
    return [f"WHERE {first_pred}"] + [((" " * FormatIndentDepth) + str(pred)) for pred in additional_preds]


def _quick_format_groupby(groupby_clause: GroupBy) -> list[str]:
    """Quick and dirty formatting logic for *GROUP BY* clauses.

    Parameters
    ----------
    groupby_clause : GroupBy
        _description_

    Returns
    -------
    list[str]
        _description_
    """
    distinct_text = "DISTINCT " if groupby_clause.distinct else ""
    if len(groupby_clause.group_columns) > 3:
        first_target, *remaining_targets = groupby_clause.group_columns
        formatted_targets = [f"GROUP BY {distinct_text}{first_target}"]
        formatted_targets += [((" " * FormatIndentDepth) + str(target)) for target in remaining_targets]
        for i in range(len(formatted_targets) - 1):
            formatted_targets[i] += ","
        return formatted_targets
    else:
        targets_text = ", ".join(str(target) for target in groupby_clause)
        return [f"GROUP BY {distinct_text}{targets_text}"]


def _quick_format_limit(limit_clause: Limit) -> list[str]:
    """Quick and dirty formatting logic for *FETCH FIRST* / *LIMIT* clauses.

    This produces output that is equivalent to the SQL standard's syntax to denote limit clauses and splits the limit
    and offset parts onto separate lines.

    Parameters
    ----------
    limit_clause : Limit
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    pass


def _subquery_replacement(expression: SqlExpression, *, inline_hints: bool,
                          indentation: int) -> SqlExpression:
    """Handler method for `transform.replace_expressions` to apply our custom `FormattingSubqueryExpression`.

    Parameters
    ----------
    expression : SqlExpression
        The expression to replace.
    inline_hints : bool
        Whether potential hint blocks should be inserted as part of the *SELECT* clause rather than before the
        actual query.
    indentation : int
        The amount of indentation to use for the subquery

    Returns
    -------
    SqlExpression
        The original SQL expression if the `expression` is not a `SubqueryExpression`. Otherwise, the expression is
        wrapped in a `FormattingSubqueryExpression`.
    """
    if not isinstance(expression, SubqueryExpression):
        return expression
    return FormattingSubqueryExpression(expression, inline_hints, indentation)


def _case_expression_replacement(expression: SqlExpression, *, indentation: int) -> SqlExpression:
    """Handler method for `transform.replace_expressions` to apply our custom `FormattingCaseExpression`.

    Parameters
    ----------
    expression : SqlExpression
        The expression to replace.
    indentation : int
        The amount of indentation to use for the case expression

    Returns
    -------
    SqlExpression
        The original SQL expression if the `expression` is not a `CaseExpression`. Otherwise, the expression is
        wrapped in a `FormattingCaseExpression`.
    """
    if not isinstance(expression, CaseExpression):
        return expression
    return FormattingCaseExpression(expression, indentation)


def _quick_format_set_query(query: SetQuery, *, inline_hint_block: bool, trailing_semicolon: bool,
                            custom_formatter: Optional[Callable[[SqlQuery], SqlQuery]]) -> str:
    """Quick and dirty formatting logic for set queries.

    Parameters
    ----------
    query : SetQuery
        The query to format
    inline_hint_block : bool
        Whether potential hint blocks should be inserted as part of the *SELECT* clause rather than before the
    trailing_semicolon : bool
        Whether to append a semicolon to the formatted query
    custom_formatter : Optional[Callable[[SqlQuery], SqlQuery]]
        A post-processing formatting service to apply to the SQL query after all preparatory steps have been performed,
        but *before* the actual formatting is started. This can be used to inject custom clause or expression
        formatting rules that are necessary to adhere to specific SQL syntax deviations for a database system. Defaults
        to *None* which skips this step.

    Returns
    -------
    str
        The pretty-printed parts of the query, indented as necessary.
    """
    # while formatting the nested queries, we still need to use rstrip in addition to trailing_semicolon=False in order to
    # format nested set queries correctly
    left_query = format_quick(query.left_query, inline_hint_block=inline_hint_block, trailing_semicolon=False,
                              custom_formatter=custom_formatter).rstrip("; ")
    right_query = format_quick(query.right_query, inline_hint_block=inline_hint_block,
                               trailing_semicolon=False, custom_formatter=custom_formatter).rstrip("; ")
    prefix = " " * FormatIndentDepth
    suffix = ";" if trailing_semicolon else ""
    return f"{left_query}\n{prefix}{query.set_operation.value}\n{right_query}{suffix}"


def format_quick(query: SelectStatement, *, inline_hint_block: bool = False, trailing_semicolon: bool = True,
                 custom_formatter: Optional[Callable[[SqlQuery], SqlQuery]] = None) -> str:
    """Applies a quick formatting heuristic to structure the given query.

    The query will be structured as follows:

    - all clauses start at a new line
    - long clauses with multiple parts (e.g. *SELECT* clause, *FROM* clause) are split along multiple intended
      lines
    - the predicate in the *WHERE* clause is split on multiple lines along the different parts of a conjunctive
      predicate

    All other clauses are written on a single line (e.g. *GROUP BY* clause).

    Parameters
    ----------
    query : SelectStatement
        The query to format
    inline_hint_block : bool, optional
        Whether to insert a potential hint block in the *SELECT* clause (i.e. *inline* it), or leave it as a
        block preceding the actual query. Defaults to *False* which indicates that the clause should be printed
        before the actual query.
    custom_formatter : Callable[[SqlQuery], SqlQuery], optional
        A post-processing formatting service to apply to the SQL query after all preparatory steps have been performed,
        but *before* the actual formatting is started. This can be used to inject custom clause or expression
        formatting rules that are necessary to adhere to specific SQL syntax deviations for a database system. Defaults
        to *None* which skips this step.

    Returns
    -------
    str
        A pretty string representation of the query.
    """
    if isinstance(query, SetQuery):
        return _quick_format_set_query(query, inline_hint_block=inline_hint_block, trailing_semicolon=trailing_semicolon,
                                       custom_formatter=custom_formatter)

    pretty_query_parts = []
    inlined_hint_block = None
    subquery_update = functools.partial(_subquery_replacement, inline_hints=inline_hint_block,
                                        indentation=FormatIndentDepth)
    case_expression_update = functools.partial(_case_expression_replacement, indentation=FormatIndentDepth)
    query = transform.replace_expressions(query, subquery_update)
    query = transform.replace_expressions(query, case_expression_update)
    if query.limit_clause is not None:
        query = transform.replace_clause(query, FormattingLimitClause(query.limit_clause))

    # Note: we cannot replace set operation clauses here, since they don't really exist in the SqlQuery object
    # instead, we have to handle them in the main loop

    if custom_formatter is not None:
        query = custom_formatter(query)

    for clause in query.clauses():
        if inline_hint_block and isinstance(clause, Hint):
            inlined_hint_block = clause
            continue

        match clause:
            case CommonTableExpression():
                pretty_query_parts.extend(_quick_format_cte(clause))
            case Select():
                pretty_query_parts.extend(_quick_format_select(clause, inlined_hint_block=inlined_hint_block))
            case ImplicitFromClause():
                pretty_query_parts.extend(_quick_format_implicit_from(clause))
            case ExplicitFromClause():
                pretty_query_parts.extend(_quick_format_explicit_from(clause))
            case From():
                pretty_query_parts.extend(_quick_format_general_from(clause))
            case Where():
                pretty_query_parts.extend(_quick_format_where(clause))
            case GroupBy():
                pretty_query_parts.extend(_quick_format_groupby(clause))
            case UnionClause() | IntersectClause() | ExceptClause():
                raise RuntimeError("Set operations should not appear in this context")
            case _:
                pretty_query_parts.append(str(clause))

    if trailing_semicolon:
        pretty_query_parts[-1] += ";"
    return "\n".join(pretty_query_parts)
