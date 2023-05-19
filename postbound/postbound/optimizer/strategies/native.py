"""Provides access to the optimization decisions made by the native database system optimizer"""
from __future__ import annotations

from typing import Optional

from postbound.db import db
from postbound.qal import qal
from postbound.optimizer import jointree
from postbound.optimizer.joinorder import enumeration


class NativeJoinOrderOptimizer(enumeration.JoinOrderOptimizer):

    def __init__(self, db_instance: db.Database) -> None:
        super().__init__()
        self.db_instance = db_instance

    def optimize_join_order(self, query: qal.SqlQuery) -> Optional[jointree.LogicalJoinTree]:
        # TODO: we could probably change the implementation at some later point in time to generate a physical QEP
        query_plan = self.db_instance.optimizer().query_plan(query)
        return jointree.LogicalJoinTree.load_from_query_plan(query_plan, query)

    def describe(self) -> dict:
        return {"name": "native", "database_system": self.db_instance.inspect()}
