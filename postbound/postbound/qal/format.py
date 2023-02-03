from __future__ import annotations

from postbound.qal import qal, clauses, predicates as preds

FORMAT_INDENT_DEPTH = 2


def _quick_format_implicit_from(from_clause: clauses.ImplicitFromClause) -> list[str]:
    tables = list(from_clause.itertables())
    if not tables:
        return []
    elif len(tables) > 3:
        first_table, *remaining_tables = tables
        return [f"FROM {first_table}"] + [((" " * FORMAT_INDENT_DEPTH) + str(tab)) for tab in remaining_tables]
    else:
        return [f"FROM {tables[0]}"]


def _quick_format_explicit_from(from_clause: clauses.ExplicitFromClause) -> list[str]:
    pretty_base = [f"FROM {from_clause.base_table}"]
    pretty_joins = [((" " * FORMAT_INDENT_DEPTH) + str(join)) for join in from_clause.joined_tables]
    return pretty_base + pretty_joins


def _quick_format_predicate(predicate: preds.AbstractPredicate) -> list[str]:
    if not isinstance(predicate, preds.CompoundPredicate):
        return [str(predicate)]
    compound_pred: preds.CompoundPredicate = predicate
    if compound_pred.operation == "and":
        first_child, *remaining_children = compound_pred.children
        return [str(first_child)] + ["AND " + str(child) for child in remaining_children]
    return [str(compound_pred)]


def _quick_format_where(where_clause: clauses.Where) -> list[str]:
    first_pred, *additional_preds = _quick_format_predicate(where_clause.predicate)
    return [f"WHERE {first_pred}"] + [((" " * FORMAT_INDENT_DEPTH) + str(pred)) for pred in additional_preds]


def format_quick(query: qal.SqlQuery) -> str:
    pretty_query_parts = []

    for clause in query.clauses():
        if isinstance(clause, clauses.ImplicitFromClause):
            pretty_query_parts.extend(_quick_format_implicit_from(clause))
        elif isinstance(clause, clauses.ExplicitFromClause):
            pretty_query_parts.extend(_quick_format_explicit_from(clause))
        elif isinstance(clause, clauses.Where):
            pretty_query_parts.extend(_quick_format_where(clause))
        else:
            pretty_query_parts.append(str(clause))

    return "\n".join(pretty_query_parts)
