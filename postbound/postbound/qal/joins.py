from __future__ import annotations

import abc

from typing import Iterable

from postbound.qal import base, predicates as preds, qal

_MospJoinTypesSQL = {join: join.upper() for join in
                     {"join", "cross join", "full join", "left join", "right join", "outer join", "inner join",
                      "natural join", "left outer join", "right outer join", "full outer join"}}


class Join(abc.ABC):
    def __init__(self, join_type: str, join_condition: preds.AbstractPredicate) -> None:
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
    def __init__(self, join_type: str, joined_table: base.TableReference,
                 join_condition: preds.AbstractPredicate = None) -> None:
        super().__init__(join_type, join_condition)
        self.joined_table = joined_table

    def is_subquery_join(self) -> bool:
        return False

    def tables(self) -> Iterable[base.TableReference]:
        return [self.joined_table]

    def __str__(self) -> str:
        join_str = _MospJoinTypesSQL.get(self.join_type, self.join_type)
        join_prefix = f"{join_str} {self.joined_table}"
        if self.join_condition:
            return join_prefix + f"ON {self.join_condition}"
        else:
            return join_prefix


class SubqueryJoin(Join):
    def __init__(self, join_type: str, subquery: qal.SqlQuery, alias: str = "",
                 join_condition: preds.AbstractPredicate = None) -> None:
        super().__init__(join_type, join_condition)
        self.subquery = subquery
        self.alias = alias

    def is_subquery_join(self) -> bool:
        return True

    def tables(self) -> Iterable[base.TableReference]:
        return self.subquery.tables()

    def __str__(self) -> str:
        join_type_str = _MospJoinTypesSQL.get(self.join_type, self.join_type)
        join_str = f"{join_type_str} ({self.subquery})"
        if self.alias:
            join_str += f" AS {self.alias}"
        if self.join_condition:
            join_str += f" ON {self.join_condition}"
        return join_str
