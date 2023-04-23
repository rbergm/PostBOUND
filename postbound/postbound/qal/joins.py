"""Models the different types of JOIN statements."""
from __future__ import annotations

import abc
import enum
from collections.abc import Iterable
from typing import Optional

from postbound.qal import base, predicates as preds, qal, expressions as expr
from postbound.util import collections as collection_utils


class JoinType(enum.Enum):
    """Indicates the actual JOIN type, e.g. OUTER JOIN or NATURAL JOIN."""
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
    """Abstract interface shared by all JOIN statements."""

    def __init__(self, join_type: JoinType, join_condition: Optional[preds.AbstractPredicate] = None,
                 *, hash_val: int) -> None:
        self._join_type = join_type
        self._join_condition = join_condition
        self._hash_val = hash_val

    @property
    def join_type(self) -> JoinType:
        """Get the kind of join to be executed."""
        return self._join_type

    @property
    def join_condition(self) -> Optional[preds.AbstractPredicate]:
        """Get the join predicate if there is one."""
        return self._join_condition

    @abc.abstractmethod
    def is_subquery_join(self) -> bool:
        """Checks, whether the JOIN statement is a table join or a subquery join."""
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced in this join."""
        raise NotImplementedError

    @abc.abstractmethod
    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are referenced in this join."""
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        """Provides access to all directly contained expressions in this join.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides access to all columns in this join.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced `n`
        times in this join, it will also be returned `n` times by this method. Furthermore, the order in which
        columns are provided by the iterable matches the order in which they appear in this join.
        """
        raise NotImplementedError

    def __hash__(self) -> int:
        return self._hash_val

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return str(self)

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError


class TableJoin(Join):
    """JOIN for a target table, e.g. SELECT * FROM R JOIN S ON R.a = S.b."""

    @staticmethod
    def inner(joined_table: base.TableReference, join_condition: preds.AbstractPredicate | None = None) -> TableJoin:
        """Constructs an INNER JOIN with the given subquery."""
        return TableJoin(joined_table, join_condition, join_type=JoinType.InnerJoin)

    def __init__(self, joined_table: base.TableReference, join_condition: Optional[preds.AbstractPredicate] = None, *,
                 join_type: JoinType = JoinType.InnerJoin) -> None:
        if not join_type or not joined_table:
            raise ValueError("Join type and table must be set")
        self._joined_table = joined_table
        hash_val = hash((join_type, joined_table, join_condition))
        super().__init__(join_type, join_condition, hash_val=hash_val)

    @property
    def joined_table(self) -> base.TableReference:
        """Get the table being joined."""
        return self._joined_table

    def is_subquery_join(self) -> bool:
        return False

    def tables(self) -> set[base.TableReference]:
        return {self.joined_table} | self.join_condition.tables()  # include joined_table just to be safe

    def columns(self) -> set[base.ColumnReference]:
        return self.join_condition.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return self.join_condition.iterexpressions()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.join_condition.itercolumns()

    __hash__ = Join.__hash__

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.joined_table == other.joined_table
                and self.join_type == other.join_type
                and self.join_condition == other.join_condition)

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
        """Constructs an INNER JOIN with the given subquery."""
        return SubqueryJoin(subquery, alias, join_condition, join_type=JoinType.InnerJoin)

    def __init__(self, subquery: qal.SqlQuery, alias: str, join_condition: Optional[preds.AbstractPredicate] = None, *,
                 join_type: JoinType = JoinType.InnerJoin) -> None:
        if not subquery or not alias:
            raise ValueError("Subquery and alias have to be set")
        self._subquery = subquery
        self._alias = alias
        hash_val = hash((join_type, subquery, alias, join_condition))
        super().__init__(join_type, join_condition, hash_val=hash_val)

    @property
    def subquery(self) -> qal.SqlQuery:
        """Get the subquery that is being joined."""
        return self._subquery

    @property
    def alias(self) -> str:
        """Get the virtual table name under which the subquery results should be made available."""
        return self._alias

    def is_subquery_join(self) -> bool:
        return True

    def target_table(self) -> base.TableReference:
        """Provides the virtual table that represents the result set of this subquery."""
        return base.TableReference.create_virtual(self._alias)

    def tables(self) -> set[base.TableReference]:
        return self.subquery.tables() | self.join_condition.tables()

    def columns(self) -> set[base.ColumnReference]:
        return self.subquery.columns() | self.join_condition.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return collection_utils.flatten([self.subquery.iterexpressions(), self.join_condition.iterexpressions()])

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten([self.subquery.itercolumns(), self.join_condition.itercolumns()])

    __hash__ = Join.__hash__

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self))
                and self.join_type == other.join_type
                and self.alias == other.alias
                and self.join_condition == other.join_condition
                and self.subquery == other.subquery)

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
