"""Provides logic to pretty-print SQL queries."""
from __future__ import annotations

from postbound.qal import qal, clauses, expressions as expr, predicates as preds

FORMAT_INDENT_DEPTH = 2


def _quick_format_select(select_clause: clauses.Select) -> list[str]:
    """Quick and dirty formatting logic for SELECT clauses.

    Up to 3 targets on the same line, otherwise one target per line.
    """
    if len(select_clause.targets) > 3:
        first_target, *remaining_targets = select_clause.targets
        formatted_targets = [f"SELECT {first_target}" if select_clause.projection_type == clauses.SelectType.Select
                             else f"SELECT DISTINCT {first_target}"]
        formatted_targets += [((" " * FORMAT_INDENT_DEPTH) + str(target)) for target in remaining_targets]
        for i in range(len(formatted_targets) - 1):
            formatted_targets += ","
        return formatted_targets
    else:
        return [str(select_clause)]


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
        return [f"FROM {tables[0]}"]


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


def format_quick(query: qal.SqlQuery) -> str:
    """Applies a quick formatting heuristic to structure the given query.

    The query will be structured as follows:

    - all clauses start at a new line
    - long clauses with multiple parts (e.g. SELECT clause, FROM clause) are split along multiple intended lines
    - the predicate in the WHERE clause is split on multiple lines along the different parts of a conjunctive predicate

    All other clauses are written on a single line (e.g. GROUP BY clause).
    """
    pretty_query_parts = []

    for clause in query.clauses():
        if isinstance(clause, clauses.Select):
            pretty_query_parts.extend(_quick_format_select(clause))
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
