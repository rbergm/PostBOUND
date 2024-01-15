from __future__ import annotations

import dataclasses
from typing import Optional

from postbound.db import db
from postbound.qal import base, qal, transform
from postbound.optimizer import joingraph, jointree


@dataclasses.dataclass(frozen=True)
class ManualJoinOrderSelection:
    query: qal.SqlQuery
    join_order: tuple[base.TableReference]
    database: db.Database

    def join_tree(self) -> jointree.LogicalJoinTree:
        base_table, *joined_tables = self.join_order
        predicates = self.query.predicates()
        base_annotation = jointree.LogicalBaseTableMetadata(predicates.filters_for(base_table))
        join_tree = jointree.LogicalJoinTree.for_base_table(base_table, base_annotation)
        for joined_table in joined_tables:
            table_annotation = jointree.LogicalBaseTableMetadata(predicates.filters_for(joined_table))
            join_annotation = jointree.LogicalJoinMetadata(predicates.joins_between(join_tree.tables(), joined_table))
            join_tree = join_tree.join_with_base_table(joined_table, table_annotation, join_annotation)
        return join_tree

    def final_query(self) -> qal.SqlQuery:
        return self.database.hinting().generate_hints(self.query, self.join_tree())


class InteractiveJoinOrderOptimizer:
    def __init__(self, query: qal.ImplicitSqlQuery, *, database: Optional[db.Database] = None) -> None:
        self._query = query
        self._db = database if database is not None else db.DatabasePool.get_instance().current_database()

    def start(self, *, use_predicate_equivalence_classes: bool = False) -> ManualJoinOrderSelection:
        join_graph = joingraph.JoinGraph(self._query, self._db.schema(),
                                         include_predicate_equivalence_classes=use_predicate_equivalence_classes)
        join_graph_stack: list[joingraph.JoinGraph] = []
        join_order: list[base.TableReference] = []
        n_tables = len(self._query.tables())

        while n_tables > len(join_order):
            intermediate_str = (" ⋈ ".join(str(tab.identifier()) for tab in join_graph.joined_tables()) if join_order
                                else "∅")
            if join_order:
                query_fragment = transform.extract_query_fragment(self._query, join_order)
                query_fragment = transform.as_count_star_query(query_fragment)
                current_card = self._db.execute_query(query_fragment)
                intermediate_str += f" (card = {current_card})"
            print("> Current intermediate:", intermediate_str)

            print("> Available actions:")
            available_joins = dict(enumerate(join_graph.available_join_paths()))
            for join_idx, join in available_joins.items():
                print(f"[{join_idx}]\tJoin {join.start_table.identifier()} ⋈ {join.target_table.identifier()}")
            print(" b\tbacktrack to last graph")
            action = input("> Select next join:")

            if action == "b":
                if len(join_graph_stack) == 0:
                    print("Already at initial join graph")
                join_graph = join_graph_stack.pop()
                join_order.pop()
                print()
                continue
            elif not action.isdigit():
                print(f"Unknown action: '{action}'\n")
                continue

            next_join_idx = int(action)
            next_join = available_joins[next_join_idx]

            join_graph_stack.append(join_graph.clone())
            if join_graph.initial():
                join_graph.mark_joined(next_join.start_table)
                join_order.append(next_join.start_table)
            join_graph.mark_joined(next_join.target_table)
            join_order.append(next_join.target_table)
            print()

        final_card = self._db.execute_query(transform.as_count_star_query(self._query))
        print(f"> Done. (final card = {final_card})")
        print("> Final join order: ", [tab.identifier() for tab in join_order])
        return ManualJoinOrderSelection(query=self._query, join_order=tuple(join_order), database=self._db)
