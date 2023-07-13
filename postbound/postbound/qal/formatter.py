"""Provides logic to pretty-print SQL queries."""
from __future__ import annotations

import functools

from typing import Optional
from postbound.qal import qal, clauses, expressions as expr, predicates as preds, transform

FORMAT_INDENT_DEPTH = 2


def _increase_indentation(content: str, indentation: Optional[int] = None) -> str:
    indentation = FORMAT_INDENT_DEPTH if indentation is None else indentation
    indent_prefix = "\n" + " " * indentation
    return " " * indentation + indent_prefix.join(content.split("\n"))


class FormattingSubqueryExpression(expr.SubqueryExpression):
    def __init__(self, original_expression: expr.SubqueryExpression, inline_hint_block: bool,
                 indentation: int) -> None:
        super().__init__(original_expression.query)
        self._inline_hint_block = inline_hint_block
        self._indentation = indentation

    def __str__(self) -> str:
        formatted = format_quick(self.query, inline_hint_block=self._inline_hint_block)
        prefix = " " * self._indentation
        if "\n" not in formatted:
            return prefix + formatted

        indented_lines = [""] + [prefix + line for line in formatted.split("\n")] + [""]
        return "\n".join(indented_lines)


def _quick_format_cte(cte_clause: clauses.CommonTableExpression) -> list[str]:
    if len(cte_clause.queries) == 1:
        cte_query = cte_clause.queries[0]
        cte_header = f"WITH {cte_query.target_name} AS ("
        cte_content = _increase_indentation(format_quick(cte_query.query).removesuffix(";"))
        cte_footer = ")"
        return [cte_header, cte_content, cte_footer]

    first_cte, *remaining_ctes = cte_clause.queries
    first_content = _increase_indentation(format_quick(first_cte.query)).removesuffix(";")
    formatted_parts: list[str] = [f"WITH {first_cte.target_name} AS (", first_content]
    for next_cte in remaining_ctes:
        current_header = f"), {next_cte.target_name} AS ("
        cte_content = _increase_indentation(format_quick(next_cte.query).removesuffix(";"))

        formatted_parts.append(current_header)
        formatted_parts.append(cte_content)

    formatted_parts.append(")")
    return formatted_parts


def _quick_format_select(select_clause: clauses.Select, *,
                         inlined_hint_block: Optional[clauses.Hint] = None) -> list[str]:
    """Quick and dirty formatting logic for SELECT clauses.

    Up to 3 targets on the same line, otherwise one target per line.
    """
    hint_text = f"{inlined_hint_block} " if inlined_hint_block else ""
    if len(select_clause.targets) > 3:
        first_target, *remaining_targets = select_clause.targets
        formatted_targets = [f"SELECT {hint_text}{first_target}"
                             if select_clause.projection_type == clauses.SelectType.Select
                             else f"SELECT DISTINCT {hint_text}{first_target}"]
        formatted_targets += [((" " * FORMAT_INDENT_DEPTH) + str(target)) for target in remaining_targets]
        for i in range(len(formatted_targets) - 1):
            formatted_targets[i] += ","
        return formatted_targets
    else:
        distinct_text = "DISTINCT " if select_clause.projection_type == clauses.SelectType.SelectDistinct else ""
        targets_text = ", ".join(str(target) for target in select_clause.targets)
        return [f"SELECT {distinct_text}{hint_text}{targets_text}"]


def _quick_format_implicit_from(from_clause: clauses.ImplicitFromClause) -> list[str]:
    """Quick and dirty formatting logic for implicit FROM clauses.

    Up to 3 tables on the same line, otherwise one table per line.
    """
    tables = list(from_clause.itertables())
    if not tables:
        return []
    elif len(tables) > 3:
        first_table, *remaining_tables = tables
        formatted_tables = [f"FROM {first_table}"]
        formatted_tables += [((" " * FORMAT_INDENT_DEPTH) + str(tab)) for tab in remaining_tables]
        for i in range(len(formatted_tables) - 1):
            formatted_tables[i] += ","
        return formatted_tables
    else:
        tables_str = ", ".join(str(tab) for tab in tables)
        return [f"FROM {tables_str}"]


def _quick_format_explicit_from(from_clause: clauses.ExplicitFromClause) -> list[str]:
    """Quick and dirty formatting logic for explicit FROM clauses: each JOIN on a different line."""
    pretty_base = [f"FROM {from_clause.base_table}"]
    pretty_joins = [((" " * FORMAT_INDENT_DEPTH) + str(join)) for join in from_clause.joined_tables]
    return pretty_base + pretty_joins


def _quick_format_predicate(predicate: preds.AbstractPredicate) -> list[str]:
    """Quick and dirty formatting logic for arbitrary (i.e. also compound) predicates.

    AND conditions on separate lines, everything else on one line.
    """
    if not isinstance(predicate, preds.CompoundPredicate):
        return [str(predicate)]
    compound_pred: preds.CompoundPredicate = predicate
    if compound_pred.operation == expr.LogicalSqlCompoundOperators.And:
        first_child, *remaining_children = compound_pred.children
        return [str(first_child)] + ["AND " + str(child) for child in remaining_children]
    return [str(compound_pred)]


def _quick_format_where(where_clause: clauses.Where) -> list[str]:
    """Quick and dirty formatting logic for WHERE clauses: one AND condition per line."""
    first_pred, *additional_preds = _quick_format_predicate(where_clause.predicate)
    return [f"WHERE {first_pred}"] + [((" " * FORMAT_INDENT_DEPTH) + str(pred)) for pred in additional_preds]


def _subquery_replacement(expression: expr.SqlExpression, *, inline_hints: bool,
                          indentation: int) -> expr.SqlExpression:
    if not isinstance(expression, expr.SubqueryExpression):
        return expression
    return FormattingSubqueryExpression(expression, inline_hints, indentation)


def format_quick(query: qal.SqlQuery, *, inline_hint_block: bool = False) -> str:
    """Applies a quick formatting heuristic to structure the given query.

    The query will be structured as follows:

    - all clauses start at a new line
    - long clauses with multiple parts (e.g. SELECT clause, FROM clause) are split along multiple intended lines
    - the predicate in the WHERE clause is split on multiple lines along the different parts of a conjunctive predicate

    All other clauses are written on a single line (e.g. GROUP BY clause).
    """
    pretty_query_parts = []
    inlined_hint_block = None
    subquery_update = functools.partial(_subquery_replacement, inline_hints=inline_hint_block,
                                        indentation=FORMAT_INDENT_DEPTH)
    query = transform.replace_expressions(query, subquery_update)

    for clause in query.clauses():
        if inline_hint_block and isinstance(clause, clauses.Hint):
            inlined_hint_block = clause
            continue

        if isinstance(clause, clauses.CommonTableExpression):
            pretty_query_parts.extend(_quick_format_cte(clause))
        elif isinstance(clause, clauses.Select):
            pretty_query_parts.extend(_quick_format_select(clause, inlined_hint_block=inlined_hint_block))
        elif isinstance(clause, clauses.ImplicitFromClause):
            pretty_query_parts.extend(_quick_format_implicit_from(clause))
        elif isinstance(clause, clauses.ExplicitFromClause):
            pretty_query_parts.extend(_quick_format_explicit_from(clause))
        elif isinstance(clause, clauses.Where):
            pretty_query_parts.extend(_quick_format_where(clause))
        else:
            pretty_query_parts.append(str(clause))

    pretty_query_parts[-1] += ";"
    return "\n".join(pretty_query_parts)
