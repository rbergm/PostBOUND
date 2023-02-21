from __future__ import annotations

import abc
import enum

from typing import Iterable

from postbound.qal import base, predicates as preds, qal


class JoinType(enum.Enum):
    InnerJoin = "JOIN"
    OuterJoin = "OUTER JOIN"
    LeftJoin = "LEFT JOIN"
    RightJoin = "RIGHT JOIN"
    CrossJoin = "CROSS JOIN"

    NaturalInnerJoin = "NATURAL JOIN"
    NaturalOuterJoin = "NATURAL OUTER JOIN"
    NaturalLeftJoin = "NATURAL LEFT JOIN"
    NaturalRightJoin = "NATURAL RIGHT JOIN"

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return self.value


class Join(abc.ABC):
    def __init__(self, join_type: JoinType, join_condition: preds.AbstractPredicate | None = None) -> None:
        self.join_type = join_type
        self.join_condition = join_condition

    @abc.abstractmethod
    def is_subquery_join(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def tables(self) -> Iterable[base.TableReference]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class TableJoin(Join):
    @staticmethod
    def inner(joined_table: base.TableReference, join_condition: preds.AbstractPredicate | None = None) -> TableJoin:
        return TableJoin(JoinType.InnerJoin, joined_table, join_condition)

    def __init__(self, join_type: JoinType, joined_table: base.TableReference,
                 join_condition: preds.AbstractPredicate | None = None) -> None:
        super().__init__(join_type, join_condition)
        self.joined_table = joined_table

    def is_subquery_join(self) -> bool:
        return False

    def tables(self) -> Iterable[base.TableReference]:
        return [self.joined_table]

    def __str__(self) -> str:
        join_str = str(self.join_type)
        join_prefix = f"{join_str} {self.joined_table}"
        if self.join_condition:
            condition_str = (f"({self.join_condition})" if self.join_condition.is_compound()
                             else str(self.join_condition))
            return join_prefix + f" ON {condition_str}"
        else:
            return join_prefix


class SubqueryJoin(Join):
    @staticmethod
    def inner(subquery: qal.SqlQuery, alias: str = "",
              join_condition: preds.AbstractPredicate | None = None) -> SubqueryJoin:
        return SubqueryJoin(JoinType.InnerJoin, subquery, alias, join_condition)

    def __init__(self, join_type: JoinType, subquery: qal.SqlQuery, alias: str = "",
                 join_condition: preds.AbstractPredicate | None = None) -> None:
        super().__init__(join_type, join_condition)
        self.subquery = subquery
        self.alias = alias

    def is_subquery_join(self) -> bool:
        return True

    def tables(self) -> Iterable[base.TableReference]:
        return self.subquery.tables()

    def __str__(self) -> str:
        join_type_str = str(self.join_type)
        join_str = f"{join_type_str} ({self.subquery})"
        if self.alias:
            join_str += f" AS {self.alias}"
        if self.join_condition:
            condition_str = (f"({self.join_condition})" if self.join_condition.is_compound()
                             else str(self.join_condition))
            join_str += f" ON {condition_str}"
        return join_str
