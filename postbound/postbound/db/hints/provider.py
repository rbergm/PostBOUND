from __future__ import annotations

import abc
import collections
from typing import Iterable

from postbound.qal import qal, base, clauses, joins, predicates, transform
from postbound.optimizer import data
from postbound.optimizer.physops import operators


class HintProvider(abc.ABC):
    @abc.abstractmethod
    def adapt_query(self, query: qal.ImplicitSqlQuery, join_order: data.JoinTree | None,
                    physical_operators: operators.PhysicalOperatorAssignment | None) -> qal.SqlQuery:
        raise NotImplementedError


def _build_subquery_alias(tables: Iterable[base.TableReference]) -> str:
    return "_".join(tab.identifier() for tab in sorted(tables))


class RightDeepJoinClauseBuilder:
    def __init__(self, query: qal.ImplicitSqlQuery) -> None:
        self.query = query
        self.available_joins: dict[base.TableReference, list[predicates.AbstractPredicate]] = (
            collections.defaultdict(list))
        self.column_renamings: dict[base.ColumnReference, str] = {}

    def for_join_tree(self, join_tree: data.JoinTree) -> clauses.ExplicitFromClause:
        self._setup()
        final_join_order = self._build_join_statements(join_tree.root)
        base_tables, *additional_joins = final_join_order
        return clauses.ExplicitFromClause(base_tables, additional_joins)

    def _setup(self) -> None:
        pass

    def _build_join_statements(self, join_node: data.JoinTreeNode) -> list[base.TableReference | joins.Join]:
        pass


class PostgresHintProvider(HintProvider):
    def adapt_query(self, query: qal.ImplicitSqlQuery, join_order: data.JoinTree | None,
                    physical_operators: operators.PhysicalOperatorAssignment | None) -> qal.SqlQuery:
        adapted_query = query
        if join_order:
            adapted_query = self._enforce_join_order(adapted_query, join_order)
        if physical_operators:
            adapted_query = self._generate_operator_hints(adapted_query, physical_operators)
        return adapted_query

    def _enforce_join_order(self, query: qal.ImplicitSqlQuery, join_order: data.JoinTree) -> qal.ExplicitSqlQuery:
        pass

    def _generate_operator_hints(self, query: qal.SqlQuery,
                                 physical_operators: operators.PhysicalOperatorAssignment) -> qal.SqlQuery:
        pass
