"""Provides logic to generate pretty strings for query objects."""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Optional

from . import transform
from ._core import (
    SqlExpression, SubqueryExpression, CaseExpression, Limit, CommonTableExpression,
    CompoundOperators,
    ImplicitFromClause, ExplicitFromClause,
    Select, Hint, Where, UnionClause, IntersectClause, ExceptClause,
    SelectType,
    AbstractPredicate, CompoundPredicate,
    SqlQuery
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
    if len(cte_clause.queries) == 1:
        cte_query = cte_clause.queries[0]
        cte_header = f"WITH {cte_query.target_name} AS ("
        cte_content = _increase_indentation(format_quick(cte_query.query, trailing_semicolon=False))
        cte_footer = ")"
        return [cte_header, cte_content, cte_footer]

    first_cte, *remaining_ctes = cte_clause.queries
    first_content = _increase_indentation(format_quick(first_cte.query, trailing_semicolon=False))
    formatted_parts: list[str] = [f"WITH {first_cte.target_name} AS (", first_content]
    for next_cte in remaining_ctes:
        current_header = f"), {next_cte.target_name} AS ("
        cte_content = _increase_indentation(format_quick(next_cte.query, trailing_semicolon=False))

        formatted_parts.append(current_header)
        formatted_parts.append(cte_content)

    formatted_parts.append(")")
    return formatted_parts


def _quick_format_select(select_clause: Select, *,
                         inlined_hint_block: Optional[Hint] = None) -> list[str]:
    """Quick and dirty formatting logic for ``SELECT`` clauses.

    Up to 3 targets are put on the same line, otherwise each target is put on a separate line.

    Parameters
    ----------
    select_clause : Select
        The clause to format
    inlined_hint_block : Optional[Hint], optional
        A hint block that should be inserted after the ``SELECT`` statement. Defaults to ``None`` which indicates that
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
    """Quick and dirty formatting logic for implicit ``FROM`` clauses.

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


def _quick_format_explicit_from(from_clause: ExplicitFromClause) -> list[str]:
    """Quick and dirty formatting logic for explicit ``FROM`` clauses.

    This function just puts each ``JOIN ON`` statement on a separate line.

    Parameters
    ----------
    from_clause : ExplicitFromClause
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    pretty_base = [f"FROM {from_clause.base_table}"]
    pretty_joins = [((" " * FormatIndentDepth) + str(join)) for join in from_clause.joined_tables]
    return pretty_base + pretty_joins


def _quick_format_predicate(predicate: AbstractPredicate) -> list[str]:
    """Quick and dirty formatting logic for arbitrary (i.e. also compound) predicates.

    ``AND`` conditions are put on separate lines, everything else is put on one line.

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
    if compound_pred.operation == CompoundOperators.And:
        first_child, *remaining_children = compound_pred.children
        return [str(first_child)] + ["AND " + str(child) for child in remaining_children]
    return [str(compound_pred)]


def _quick_format_where(where_clause: Where) -> list[str]:
    """Quick and dirty formatting logic for ``WHERE`` clauses.

    This function just puts each part of an ``AND`` condition on a separate line and leaves the parts of ``OR``
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


def _quick_format_limit(limit_clause: Limit) -> list[str]:
    """Quick and dirty formatting logic for ``FETCH FIRST`` / ``LIMIT`` clauses.

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


def _quick_format_union(union_clause: UnionClause) -> list[str]:
    """Quick and dirty formatting logic for ``UNION`` clauses.

    This function just puts each part of the union query on a separate line.

    Parameters
    ----------
    union_clause : UnionClause
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    prefix = [" UNION ALL"] if union_clause.is_union_all() else [" UNION"]
    formatted_query = format_quick(union_clause.query, trailing_semicolon=False)
    lines = formatted_query.split("\n")
    return prefix + lines


def _quick_format_intersect(intersect_clause: IntersectClause) -> list[str]:
    """Quick and dirty formatting logic for ``INTERSECT`` clauses.

    This function just puts each part of the intersect query on a separate line.

    Parameters
    ----------
    intersect_clause : IntersectClause
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    prefix = [" INTERSECT"]
    formatted_query = format_quick(intersect_clause.query, trailing_semicolon=False)
    lines = formatted_query.split("\n")
    return prefix + lines


def _quick_format_except(except_clause: ExceptClause) -> list[str]:
    """Quick and dirty formatting logic for ``EXCEPT`` clauses.

    This function just puts each part of the except query on a separate line.

    Parameters
    ----------
    except_clause : ExceptClause
        The clause to format

    Returns
    -------
    list[str]
        The pretty-printed parts of the clause, indented as necessary.
    """
    prefix = [" EXCEPT"]
    formatted_query = format_quick(except_clause.query, trailing_semicolon=False)
    lines = formatted_query.split("\n")
    return prefix + lines


def _subquery_replacement(expression: SqlExpression, *, inline_hints: bool,
                          indentation: int) -> SqlExpression:
    """Handler method for `transform.replace_expressions` to apply our custom `FormattingSubqueryExpression`.

    Parameters
    ----------
    expression : SqlExpression
        The expression to replace.
    inline_hints : bool
        Whether potential hint blocks should be inserted as part of the ``SELECT`` clause rather than before the
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


def format_quick(query: SqlQuery, *, inline_hint_block: bool = False, trailing_semicolon: bool = True,
                 custom_formatter: Optional[Callable[[SqlQuery], SqlQuery]] = None) -> str:
    """Applies a quick formatting heuristic to structure the given query.

    The query will be structured as follows:

    - all clauses start at a new line
    - long clauses with multiple parts (e.g. ``SELECT`` clause, ``FROM`` clause) are split along multiple intended
      lines
    - the predicate in the ``WHERE`` clause is split on multiple lines along the different parts of a conjunctive
      predicate

    All other clauses are written on a single line (e.g. ``GROUP BY`` clause).

    Parameters
    ----------
    query : SqlQuery
        The query to format
    inline_hint_block : bool, optional
        Whether to insert a potential hint block in the ``SELECT`` clause (i.e. *inline* it), or leave it as a
        block preceding the actual query. Defaults to ``False`` which indicates that the clause should be printed
        before the actual query.
    custom_formatter : Callable[[SqlQuery], SqlQuery], optional
        A post-processing formatting service to apply to the SQL query after all preparatory steps have been performed,
        but *before* the actual formatting is started. This can be used to inject custom clause or expression
        formatting rules that are necessary to adhere to specific SQL syntax deviations for a database system. Defaults
        to ``None`` which skips this step.

    Returns
    -------
    str
        A pretty string representation of the query.
    """
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

        if isinstance(clause, CommonTableExpression):
            pretty_query_parts.extend(_quick_format_cte(clause))
        elif isinstance(clause, Select):
            pretty_query_parts.extend(_quick_format_select(clause, inlined_hint_block=inlined_hint_block))
        elif isinstance(clause, ImplicitFromClause):
            pretty_query_parts.extend(_quick_format_implicit_from(clause))
        elif isinstance(clause, ExplicitFromClause):
            pretty_query_parts.extend(_quick_format_explicit_from(clause))
        elif isinstance(clause, Where):
            pretty_query_parts.extend(_quick_format_where(clause))
        elif isinstance(clause, UnionClause):
            pretty_query_parts.extend(_quick_format_union(clause))
        elif isinstance(clause, IntersectClause):
            pretty_query_parts.extend(_quick_format_intersect(clause))
        elif isinstance(clause, ExceptClause):
            pretty_query_parts.extend(_quick_format_except(clause))
        else:
            pretty_query_parts.append(str(clause))

    if trailing_semicolon:
        pretty_query_parts[-1] += ";"
    return "\n".join(pretty_query_parts)
