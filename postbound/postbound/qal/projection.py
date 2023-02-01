from __future__ import annotations

from dataclasses import dataclass

from postbound.qal import expressions as expr


@dataclass
class BaseProjection:
    expression: expr.SqlExpression
    target_name: str = ""

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
        pass

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        select_str = _MospSelectTypesSQL.get(self.projection_type, self.projection_type)
        parts_str = ", ".join(str(target) for target in self.targets)
        return f"{select_str} {parts_str}"
