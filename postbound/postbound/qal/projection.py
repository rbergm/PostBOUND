from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from postbound.qal import base, expressions as expr
from postbound.util import collections as collection_utils


@dataclass
class BaseProjection:
    expression: expr.SqlExpression
    target_name: str = ""

    def columns(self) -> set[base.ColumnReference]:
        return self.expression.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.expression.itercolumns()

    def tables(self) -> set[base.TableReference]:
        return self.expression.tables()

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if not self.target_name:
            return str(self.expression)
        return f"{self.expression} AS {self.target_name}"


_MospSelectTypesSQL = {"select": "SELECT", "select_distinct": "SELECT DISTINCT"}


class QueryProjection:
    def __init__(self, targets: list[BaseProjection], projection_type: str = "select") -> None:
        self.targets = targets
        self.projection_type = projection_type

    def parts(self) -> list[BaseProjection]:
        return self.targets

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(target.columns() for target in self.targets)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(target.itercolumns() for target in self.targets)

    def tables(self) -> set[base.TableReference]:
        return {target.tables() for target in self.targets}

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        select_str = _MospSelectTypesSQL.get(self.projection_type, self.projection_type)
        parts_str = ", ".join(str(target) for target in self.targets)
        return f"{select_str} {parts_str}"


def count_star() -> BaseProjection:
    return BaseProjection(expr.FunctionExpression("count", [expr.StarExpression()]))
