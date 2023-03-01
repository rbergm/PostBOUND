from __future__ import annotations

import abc
import enum

from dataclasses import dataclass
from typing import Iterable

from postbound.qal import base, expressions as expr, joins, predicates as preds
from postbound.util import collections as collection_utils


@dataclass
class BaseProjection:
    expression: expr.SqlExpression
    target_name: str = ""

    @staticmethod
    def count_star() -> BaseProjection:
        return BaseProjection(expr.FunctionExpression("count", [expr.StarExpression()]))

    @staticmethod
    def star() -> BaseProjection:
        return BaseProjection(expr.StarExpression())

    @staticmethod
    def column(col: base.ColumnReference, target_name: str = "") -> BaseProjection:
        return BaseProjection(expr.ColumnExpression(col), target_name)

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


@dataclass
class Hint:
    preparatory_statements: str = ""
    query_hints: str = ""

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.preparatory_statements and self.query_hints:
            return self.preparatory_statements + "\n" + self.query_hints
        elif self.preparatory_statements:
            return self.preparatory_statements
        return self.query_hints


@dataclass
class Explain:
    analyze: bool = False
    format: str | None = None

    @staticmethod
    def explain_analyze(format_type: str = "JSON") -> Explain:
        return Explain(True, format_type)

    @staticmethod
    def pure(format_type: str = "JSON") -> Explain:
        return Explain(False, format_type)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        explain_prefix = "EXPLAIN"
        explain_body = ""
        if self.analyze and self.format:
            explain_body = f" (ANALYZE, FORMAT {self.format})"
        elif self.analyze:
            explain_body = " ANALYZE"
        elif self.format:
            explain_body = f" (FORMAT {self.format})"
        return explain_prefix + explain_body


class SelectType(enum.Enum):
    Select = "SELECT"
    SelectDistinct = "SELECT DISTINCT"


class Select:
    def __init__(self, targets: BaseProjection | list[BaseProjection],
                 projection_type: SelectType = SelectType.Select) -> None:
        self.targets = collection_utils.enlist(targets)
        self.projection_type = projection_type

    def parts(self) -> list[BaseProjection]:
        return self.targets

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(target.columns() for target in self.targets)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(target.itercolumns() for target in self.targets)

    def tables(self) -> set[base.TableReference]:
        return collection_utils.set_union(target.tables() for target in self.targets)

    def output_names(self) -> dict[str, base.ColumnReference]:
        output = {}
        for projection in self.targets:
            if not projection.target_name:
                continue
            source_columns = projection.expression.columns()
            if len(source_columns) != 1:
                continue
            output[projection.target_name] = collection_utils.simplify(source_columns)
        return output

    def __hash__(self) -> int:
        return hash((self.projection_type, tuple(self.targets)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.projection_type == other.projection_type
                and self.targets == other.targets)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        select_str = self.projection_type.value
        parts_str = ", ".join(str(target) for target in self.targets)
        return f"{select_str} {parts_str}"


class From(abc.ABC):
    @abc.abstractmethod
    def tables(self) -> set[base.TableReference]:
        raise NotImplementedError

    @abc.abstractmethod
    def predicates(self) -> preds.QueryPredicates | None:
        raise NotImplementedError

    @abc.abstractmethod
    def __hash__(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def __eq__(self, other) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class ImplicitFromClause(From):
    def __init__(self, tables: base.TableReference | list[base.TableReference] | None = None):
        self._tables = collection_utils.enlist(tables) if tables is not None else []

    def tables(self) -> set[base.TableReference]:
        return set(self._tables)

    def itertables(self) -> Iterable[base.TableReference]:
        return self._tables

    def predicates(self) -> preds.QueryPredicates | None:
        return None

    def __hash__(self) -> int:
        return hash(tuple(self._tables))

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._tables == other._tables

    def __str__(self) -> str:
        if not self._tables:
            return "[NO TABLES]"
        return "FROM " + ", ".join(str(table) for table in self._tables)


class ExplicitFromClause(From):
    def __init__(self, base_table: base.TableReference, joined_tables: list[joins.Join]):
        self.base_table = base_table
        self.joined_tables = joined_tables

    def tables(self) -> set[base.TableReference]:
        all_tables = [self.base_table]
        for join in self.joined_tables:
            all_tables.extend(join.tables())
        return set(all_tables)

    def predicates(self) -> preds.QueryPredicates | None:
        all_predicates = preds.QueryPredicates.empty_predicate()
        for join in self.joined_tables:
            if isinstance(join, joins.TableJoin):
                if join.join_condition:
                    all_predicates = all_predicates.and_(join.join_condition)
                continue

            if not isinstance(join, joins.SubqueryJoin):
                TypeError("Unknown join type: " + str(type(join)))
            subquery_join: joins.SubqueryJoin = join

            subquery_predicates = subquery_join.subquery.predicates()
            if subquery_predicates:
                all_predicates = all_predicates.and_(subquery_predicates)
            if subquery_join.join_condition:
                all_predicates = all_predicates.and_(subquery_join.join_condition)

        return all_predicates

    def __hash__(self) -> int:
        return hash((self.base_table, tuple(self.joined_tables)))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.base_table == other.base_table
                and self.joined_tables == other.joined_tables)

    def __str__(self) -> str:
        return f"FROM {self.base_table} " + " ".join(str(join) for join in self.joined_tables)


@dataclass
class Where:
    predicate: preds.AbstractPredicate

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"WHERE {self.predicate}"


@dataclass
class GroupBy:
    group_columns: list[expr.SqlExpression]
    distinct: bool = False

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        columns_str = ", ".join(str(col) for col in self.group_columns)
        distinct_str = " DISTINCT" if self.distinct else ""
        return f"GROUP BY{distinct_str} {columns_str}"


@dataclass
class Having:
    condition: preds.AbstractPredicate

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"HAVING {self.condition}"


@dataclass
class OrderByExpression:
    column: expr.SqlExpression
    ascending: bool | None = None
    nulls_first: bool | None = None

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        ascending_str = "" if self.ascending is None else (" ASC" if self.ascending else " DESC")
        nulls_first = "" if self.nulls_first is None else (" NULLS FIRST " if self.nulls_first else " NULLS LAST")
        return f"{self.column}{ascending_str}{nulls_first}"


@dataclass
class OrderBy:
    expressions: list[OrderByExpression]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "ORDER BY " + ", ".join(str(order_expr) for order_expr in self.expressions)


class Limit:
    def __init__(self, *, limit: int | None = None, offset: int | None = None) -> None:
        if limit is None and offset is None:
            raise ValueError("LIMIT and OFFSET cannot be both unspecified")
        self.limit = limit
        self.offset = offset

    def __hash__(self) -> int:
        return hash((self.limit, self.offset))

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.limit == other.limit and self.offset == other.offset

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        limit_str = f"LIMIT {self.limit}" if self.limit is not None else ""
        offset_str = f"OFFSET {self.offset}" if self.offset is not None else ""
        return limit_str + offset_str
