"""Contains the implementation of all supported SQL clauses."""
from __future__ import annotations

import abc
import enum
from collections.abc import Sequence
from typing import Iterable, Optional

from postbound.qal import base, expressions as expr, qal, predicates as preds
from postbound.util import collections as collection_utils


class BaseClause(abc.ABC):
    """Basic interface shared by all supported clauses. This is an abstract interface, not a usable clause."""

    def __init__(self, hash_val: int):
        self._hash_val = hash_val

    def tables(self) -> set[base.TableReference]:
        """Provides all tables that are referenced in the clause."""
        return {column.table for column in self.columns() if column.is_bound()}

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        """Provides all columns that are referenced in the clause."""
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        """Provides access to all directly contained expressions in this clause.

        Nested expressions can be accessed from these expressions in a recursive manner (see the `SqlExpression`
        interface for details).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[base.ColumnReference]:
        """Provides access to all column in this clause.

        In contrast to the `columns` method, duplicates are returned multiple times, i.e. if a column is referenced `n`
        times in this clause, it will also be returned `n` times by this method. Furthermore, the order in which
        columns are provided by the iterable matches the order in which they appear in this clause.
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


class Hint(BaseClause):
    """Hint block of a clause.

    Depending on the SQL dialect, these hints will be placed at different points in the query. Furthermore, the precise
    contents (i.e. syntax and semantic) vary from database system to system.

    Hints are differentiated in two parts:

    - preparatory statements can be executed as valid commands on the database system, e.g. optimizer settings, etc.
    - query hints are the actual hints. Typically, these will be inserted as comments at some place in the query.

    For example, a hint clause for MySQL could look like this:

    ```sql
    SET optimizer_switch = 'block_nested_loop=off';
    SELECT /*+ HASH_JOIN(R S) */ R.a
    FROM R, S, T
    WHERE R.a = S.b AND S.b = T.c
    ```

    This enforces the join between tables `R` and `S` to be executed as a hash join (due to the query hint) and
    disables usage of the block nested-loop join for the entire query (which in this case only affects the join between
    tables `S` and `T`) due to the preparatory `SET optimizer_switch` statement.
    """

    def __init__(self, preparatory_statements: str = "", query_hints: str = ""):
        self._preparatory_statements = preparatory_statements
        self._query_hints = query_hints

        hash_val = hash((preparatory_statements, query_hints))
        super().__init__(hash_val)

    @property
    def preparatory_statements(self) -> str:
        """Get the string of preparatory statements. Can be empty."""
        return self._preparatory_statements

    @property
    def query_hints(self) -> str:
        """Get the query hint text. Can be empty."""
        return self._query_hints

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.preparatory_statements == other.preparatory_statements
                and self.query_hints == other.query_hints)

    def __str__(self) -> str:
        if self.preparatory_statements and self.query_hints:
            return self.preparatory_statements + "\n" + self.query_hints
        elif self.preparatory_statements:
            return self.preparatory_statements
        return self.query_hints


class Explain(BaseClause):
    """EXPLAIN block of a query.

    EXPLAIN queries change the execution mode of a query. Instead of focusing on the actual query result, an EXPLAIN
    query produces information about the internal processes of the database system. Typically, this includes which
    execution plan the DBS would choose for the query. Additionally, EXPLAIN ANALYZE (as for example supported by
    Postgres) provides the query plan and executes the actual query. The returned plan is then annotated by how the
    optimizer predictions match reality. Furthermore, such ANALYZE plans typically also contain some runtime statistics
    such as runtime of certain operators.

    The precise syntax and semantic of an EXPLAIN statement depends on the actual DBS. The Explain clause object
    is modeled after Postgres.
    """

    @staticmethod
    def explain_analyze(format_type: str = "JSON") -> Explain:
        """Constructs an EXPLAIN ANALYZE clause with the specified output format."""
        return Explain(True, format_type)

    @staticmethod
    def plan(format_type: str = "JSON") -> Explain:
        """Constructs a pure EXPLAIN clause (i.e. without ANALYZE) with the specified output format."""
        return Explain(False, format_type)

    def __init__(self, analyze: bool = False, target_format: Optional[str] = None):
        self._analyze = analyze
        self._target_format = target_format

        hash_val = hash((analyze, target_format))
        super().__init__(hash_val)

    @property
    def analyze(self) -> bool:
        """Check, whether the query should be executed as EXPLAIN ANALYZE rather than just plain EXPLAIN.

        Usually, EXPLAIN ANALYZE executes the query and gathers extensive runtime statistics (e.g. comparing estimated
        vs. true cardinalities for intermediate nodes).
        """
        return self._analyze

    @property
    def target_format(self) -> Optional[str]:
        """Get the target format in which the EXPLAIN plan should be provided."""
        return self._target_format

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.analyze == other.analyze
                and self.target_format == other.target_format)

    def __str__(self) -> str:
        explain_prefix = "EXPLAIN"
        explain_body = ""
        if self.analyze and self.target_format:
            explain_body = f" (ANALYZE, FORMAT {self.target_format})"
        elif self.analyze:
            explain_body = " ANALYZE"
        elif self.target_format:
            explain_body = f" (FORMAT {self.target_format})"
        return explain_prefix + explain_body


class BaseProjection:
    """The `BaseProjection` forms the fundamental ingredient for a SELECT clause.

    Each SELECT clause is composed of at least one `BaseProjection`. Each projection can be an arbitrary
    `SqlExpression` (rules and restrictions of the SQL standard are not enforced here). In addition, each projection
    can receive a target name as in `SELECT foo AS f FROM bar`.
    """

    @staticmethod
    def count_star() -> BaseProjection:
        """Shortcut to create a COUNT(*) projection."""
        return BaseProjection(expr.FunctionExpression("count", [expr.StarExpression()]))

    @staticmethod
    def star() -> BaseProjection:
        """Shortcut to create a * projection."""
        return BaseProjection(expr.StarExpression())

    @staticmethod
    def column(col: base.ColumnReference, target_name: str = "") -> BaseProjection:
        """Shortcut to create a projection for the given column."""
        return BaseProjection(expr.ColumnExpression(col), target_name)

    def __init__(self, expression: expr.SqlExpression, target_name: str = ""):
        if not expression:
            raise ValueError("Expression must be set")
        self._expression = expression
        self._target_name = target_name
        self._hash_val = hash((expression, target_name))

    @property
    def expression(self) -> expr.SqlExpression:
        """Get the expression that forms the column."""
        return self._expression

    @property
    def target_name(self) -> str:
        """Get the alias under which the column should be accessible. Can be empty."""
        return self._target_name

    def columns(self) -> set[base.ColumnReference]:
        return self.expression.columns()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.expression.itercolumns()

    def tables(self) -> set[base.TableReference]:
        return self.expression.tables()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.expression == other.expression and self.target_name == other.target_name)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if not self.target_name:
            return str(self.expression)
        return f"{self.expression} AS {self.target_name}"


class SelectType(enum.Enum):
    """Indicates the specific type of the SELECT clause."""
    Select = "SELECT"
    SelectDistinct = "SELECT DISTINCT"


class Select(BaseClause):
    """The SELECT clause of a query.

    This is the only required part of a query. Everything else is optional and can be left out. (Notice that PostBOUND
    is focused on SPJ-queries, hence there are no INSERT, UPDATE, or DELETE queries)

    A SELECT clause simply consists of a number of individual projections (see `BaseProjection`), the `targets`.
    """

    @staticmethod
    def count_star() -> Select:
        """Shortcut to create a SELECT COUNT(*) clause."""
        return Select(BaseProjection.count_star())

    @staticmethod
    def star() -> Select:
        """Shortcut to create a SELECT * clause."""
        return Select(BaseProjection.star())

    def create_for(columns: Iterable[base.ColumnReference],
                   projection_type: SelectType = SelectType.Select) -> Select:
        target_columns = [BaseProjection.column(column) for column in columns]
        return Select(target_columns, projection_type)

    def __init__(self, targets: BaseProjection | Sequence[BaseProjection],
                 projection_type: SelectType = SelectType.Select) -> None:
        if not targets:
            raise ValueError("At least one target must be specified")
        self._targets = tuple(collection_utils.enlist(targets))
        self._projection_type = projection_type

        hash_val = hash((self._projection_type, self._targets))
        super().__init__(hash_val)

    @property
    def targets(self) -> Sequence[BaseProjection]:
        """Get all projections."""
        return self._targets

    @property
    def projection_type(self) -> SelectType:
        """Get the type of projection (with or without duplicate elimination)."""
        return self._projection_type

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(target.columns() for target in self.targets)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(target.itercolumns() for target in self.targets)

    def tables(self) -> set[base.TableReference]:
        return collection_utils.set_union(target.tables() for target in self.targets)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [target.expression for target in self.targets]

    def output_names(self) -> dict[str, base.ColumnReference]:
        """Output names map the alias of each column to the actual column.

        For example, consider a query `SELECT R.a AS foo, R.b AS bar FROM R`. Calling `output_names` on this query
        provides the dictionary `{'foo': R.a, 'bar': R.b}`.

        Currently, this method only works for 1:1 mappings and other aliases are ignored. For example, consider a query
        `SELECT my_udf(R.a, R.b) AS c FROM R`. Here, a user-defined function is used to combine the values of `R.a` and
        `R.b` to form an output column `c`. Such a projection is ignored by `output_names`.
        """
        output = {}
        for projection in self.targets:
            if not projection.target_name:
                continue
            source_columns = projection.expression.columns()
            if len(source_columns) != 1:
                continue
            output[projection.target_name] = collection_utils.simplify(source_columns)
        return output

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.projection_type == other.projection_type
                and self.targets == other.targets)

    def __str__(self) -> str:
        select_str = self.projection_type.value
        parts_str = ", ".join(str(target) for target in self.targets)
        return f"{select_str} {parts_str}"


class TableSource(abc.ABC):
    @abc.abstractmethod
    def tables(self) -> set[base.TableReference]:
        raise NotImplementedError

    @abc.abstractmethod
    def columns(self) -> set[base.ColumnReference]:
        raise NotImplementedError

    @abc.abstractmethod
    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        raise NotImplementedError

    @abc.abstractmethod
    def itercolumns(self) -> Iterable[base.ColumnReference]:
        raise NotImplementedError

    def predicates(self) -> preds.QueryPredicates | None:
        raise NotImplementedError


class DirectTableSource(TableSource):
    def __init__(self, table: base.TableReference) -> None:
        self._table = table

    @property
    def table(self) -> base.TableReference:
        return self._table

    def tables(self) -> set[base.TableReference]:
        return {self._table}

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    def predicates(self) -> preds.QueryPredicates | None:
        return None

    def __hash__(self) -> int:
        return hash(self._table)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self._table == other._table

    def __repr__(self) -> str:
        return str(self._table)

    def __str__(self) -> str:
        return str(self._table)


class SubqueryTableSource(TableSource):

    def __init__(self, query: qal.SqlQuery | expr.SubqueryExpression, target_name: str) -> None:
        self._subquery_expression = (query if isinstance(query, expr.SubqueryExpression)
                                     else expr.SubqueryExpression(query))
        self._target_name = target_name
        self._hash_val = hash((self._subquery_expression, self._target_name))

    @property
    def query(self) -> qal.SqlQuery:
        return self._subquery_expression.query

    @property
    def target_name(self) -> str:
        return self._target_name

    @property
    def expression(self) -> expr.SubqueryExpression:
        return self._subquery_expression

    def tables(self) -> set[base.TableReference]:
        return self._subquery_expression.tables()

    def columns(self) -> set[base.ColumnReference]:
        return self._subquery_expression.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [self._subquery_expression]

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self._subquery_expression.itercolumns()

    def predicates(self) -> preds.QueryPredicates | None:
        return self._subquery_expression.query.predicates()

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self)) and self._subquery_expression == other._subquery_expression
                and self._target_name == other._target_name)

    def __repr__(self) -> str:
        return str(self._subquery_expression)

    def __str__(self) -> str:
        query_str = str(self._subquery_expression.query).removesuffix(";")
        return f"({query_str}) AS {self._target_name}"


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


class JoinTableSource(TableSource):
    def __init__(self, source: TableSource, join_condition: Optional[preds.AbstractPredicate] = None, *,
                 join_type: JoinType = JoinType.InnerJoin) -> None:
        if isinstance(source, JoinTableSource):
            raise ValueError("JOIN statements cannot have another JOIN statement as source")
        self._source = source
        self._join_condition = join_condition
        self._join_type = join_type if join_condition else JoinType.CrossJoin
        self._hash_val = hash((self._source, self._join_condition, self._join_type))

    @property
    def source(self) -> TableSource:
        return self._source

    @property
    def join_condition(self) -> Optional[preds.AbstractPredicate]:
        return self._join_condition

    @property
    def join_type(self) -> JoinType:
        return self._join_type

    def tables(self) -> set[base.TableReference]:
        return self._source.tables()

    def columns(self) -> set[base.ColumnReference]:
        condition_columns = self._join_condition.columns() if self._join_condition else set()
        return self._source.columns() | condition_columns

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        source_expressions = list(self._source.iterexpressions())
        condition_expressions = list(self._join_condition.iterexpressions()) if self._join_condition else []
        return source_expressions + condition_expressions

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        source_columns = list(self._source.itercolumns())
        condition_columns = list(self._join_condition.itercolumns()) if self._join_condition else []
        return source_columns + condition_columns

    def predicates(self) -> preds.QueryPredicates | None:
        source_predicates = self._source.predicates()
        condition_predicates = preds.QueryPredicates(self._join_condition) if self._join_condition else None

        if source_predicates and condition_predicates:
            return source_predicates.and_(condition_predicates)
        elif source_predicates:
            return source_predicates
        elif condition_predicates:
            return condition_predicates
        else:
            return None

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other: object) -> bool:
        return (isinstance(other, type(self)) and self._source == other._source
                and self._join_condition == other._join_condition
                and self._join_type == other._join_type)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        join_str = str(self.join_type)
        join_prefix = f"{join_str} {self.source}"
        if self.join_condition:
            condition_str = (f"({self.join_condition})" if self.join_condition.is_compound()
                             else str(self.join_condition))
            return join_prefix + f" ON {condition_str}"
        else:
            return join_prefix


class From(BaseClause):
    def __init__(self, contents: TableSource | Iterable[TableSource]):
        self._contents: tuple[TableSource] = tuple(collection_utils.enlist(contents))
        super().__init__(hash(self._contents))

    @property
    def contents(self) -> Sequence[TableSource]:
        return self._contents

    def tables(self) -> set[base.TableReference]:
        return collection_utils.set_union(src.tables() for src in self._contents)

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(src.columns() for src in self._contents)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return collection_utils.flatten(src.iterexpressions() for src in self._contents)

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(src.itercolumns() for src in self._contents)

    def predicates(self) -> preds.QueryPredicates | None:
        source_predicates = [src.predicates() for src in self._contents]
        if not any(source_predicates):
            return None
        actual_predicates = [src_pred.root() for src_pred in source_predicates if src_pred]
        merged_predicate = preds.CompoundPredicate.create_and(actual_predicates)
        return preds.QueryPredicates(merged_predicate)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self._contents == other._contents

    def __str__(self) -> str:
        fixture = "FROM "
        contents_str = []
        for src in self._contents:
            if isinstance(src, JoinTableSource):
                contents_str.append(" " + str(src))
            elif contents_str:
                contents_str.append(", " + str(src))
            else:
                contents_str.append(str(src))
        return fixture + "".join(contents_str)


class ImplicitFromClause(From):

    @staticmethod
    def create_for(tables: base.TableReference | Iterable[base.TableReference]) -> ImplicitFromClause:
        tables = collection_utils.enlist(tables)
        return ImplicitFromClause([DirectTableSource(tab) for tab in tables])

    def __init__(self, tables: DirectTableSource | Iterable[DirectTableSource] | None = None):
        super().__init__(tables)

    def itertables(self) -> Sequence[base.TableReference]:
        return [src.table for src in self.contents]


class ExplicitFromClause(From):
    """The explicit FROM clause lists all referenced tables and subqueries using the JOIN ON syntax.

    See `FromClause` for details.
    """

    def __init__(self, base_table: DirectTableSource | SubqueryTableSource, joined_tables: Iterable[JoinTableSource]):
        super().__init__([base_table] + list(joined_tables))
        self._base_table = base_table
        self._joined_tables = tuple(joined_tables)
        if not self._joined_tables:
            raise ValueError("At least one joined table expected!")

    @property
    def base_table(self) -> DirectTableSource | SubqueryTableSource:
        """Get the first table that is part of the FROM clause."""
        return self._base_table

    @property
    def joined_tables(self) -> Sequence[JoinTableSource]:
        """Get all tables that are defined in the JOIN ON syntax."""
        return self._joined_tables


class Where(BaseClause):
    """The WHERE clause specifies conditions that result rows must satisfy.

    All conditions are collected in a (potentially conjunctive or disjunctive) predicate object. See
    `AbstractPredicate` for details.
    """

    def __init__(self, predicate: preds.AbstractPredicate) -> None:
        if not predicate:
            raise ValueError("Predicate must be set")
        self._predicate = predicate
        super().__init__(hash(predicate))

    @property
    def predicate(self) -> preds.AbstractPredicate:
        """Get the root predicate that contains all filters and joins in the WHERE clause."""
        return self._predicate

    def columns(self) -> set[base.ColumnReference]:
        return self.predicate.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return self.predicate.iterexpressions()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.predicate.itercolumns()

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.predicate == other.predicate

    def __str__(self) -> str:
        return f"WHERE {self.predicate}"


class GroupBy(BaseClause):
    """The GROUP BY clause combines rows that match a grouping criterion to enable aggregation on these groups.

    All grouped columns can be arbitrary `SqlExpression`s, rules and restrictions of the SQL standard are not enforced
    by PostBOUND.
    """

    def __init__(self, group_columns: Sequence[expr.SqlExpression], distinct: bool = False) -> None:
        if not group_columns:
            raise ValueError("At least one group column must be specified")
        self._group_columns = tuple(group_columns)
        self._distinct = distinct

        hash_val = hash((self._group_columns, self._distinct))
        super().__init__(hash_val)

    @property
    def group_columns(self) -> Sequence[expr.SqlExpression]:
        """Get all expressions that should be used to determine the grouping."""
        return self._group_columns

    @property
    def distinct(self) -> bool:
        """Get whether the grouping should eliminate duplicates."""
        return self._distinct

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(column.columns() for column in self.group_columns)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return self.group_columns

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(column.itercolumns() for column in self.group_columns)

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.group_columns == other.group_columns and self.distinct == other.distinct)

    def __str__(self) -> str:
        columns_str = ", ".join(str(col) for col in self.group_columns)
        distinct_str = " DISTINCT" if self.distinct else ""
        return f"GROUP BY{distinct_str} {columns_str}"


class Having(BaseClause):
    """The HAVING clause specifies conditions that have to be met on the groups constructed by a GROUP BY clause.

    All conditions are collected in a (potentially conjunctive or disjunctive) predicate object. See
    `AbstractPredicate` for details.
    """

    def __init__(self, condition: preds.AbstractPredicate) -> None:
        if not condition:
            raise ValueError("Condition must be set")
        self._condition = condition
        super().__init__(hash(condition))

    @property
    def condition(self) -> preds.AbstractPredicate:
        """Get the root predicate that is used to form the HAVING clause."""
        return self._condition

    def columns(self) -> set[base.ColumnReference]:
        return self.condition.columns()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return self.condition.iterexpressions()

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return self.condition.itercolumns()

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.condition == other.condition

    def __str__(self) -> str:
        return f"HAVING {self.condition}"


class OrderByExpression:
    """The `OrderByExpression` is the fundamental ingredient for an ORDER BY clause.

    Each expression consists of the actual column (which might be an arbitrary `SqlExpression`, rules and restrictions
    by the SQL standard are not enforced here) as well as information regarding the ordering of the column. Setting
    such information to `None` falls back to the default interpretation by the target database system.
    """

    def __init__(self, column: expr.SqlExpression, ascending: Optional[bool] = None,
                 nulls_first: Optional[bool] = None) -> None:
        if not column:
            raise ValueError("Column must be specified")
        self._column = column
        self._ascending = ascending
        self._nulls_first = nulls_first
        self._hash_val = hash((self._column, self._ascending, self._nulls_first))

    @property
    def column(self) -> expr.SqlExpression:
        """Get the expression used to specify the current grouping."""
        return self._column

    @property
    def ascending(self) -> bool:
        """Get the desired ordering of the output rows."""
        return self._ascending

    @property
    def nulls_first(self) -> bool:
        """Get where to place NULL values in the result set."""
        return self._nulls_first

    def __hash__(self) -> int:
        return self._hash_val

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and self.column == other.column
                and self.ascending == other.ascending
                and self.nulls_first == other.nulls_first)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        ascending_str = "" if self.ascending is None else (" ASC" if self.ascending else " DESC")
        nulls_first = "" if self.nulls_first is None else (" NULLS FIRST " if self.nulls_first else " NULLS LAST")
        return f"{self.column}{ascending_str}{nulls_first}"


class OrderBy(BaseClause):
    """The ORDER BY clause specifies how result rows should be sorted.

    It consists of an arbitrary number of `OrderByExpression`s.
    """

    def __init__(self, expressions: list[OrderByExpression]) -> None:
        if not expressions:
            raise ValueError("At least one ORDER BY expression required")
        self._expressions = tuple(expressions)
        super().__init__(hash(self._expressions))

    @property
    def expressions(self) -> Sequence[OrderByExpression]:
        """Get the expressions that form this ORDER BY clause."""
        return self._expressions

    def columns(self) -> set[base.ColumnReference]:
        return collection_utils.set_union(expression.column.columns() for expression in self.expressions)

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return [expression.column for expression in self.expressions]

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return collection_utils.flatten(expression.itercolumns() for expression in self.iterexpressions())

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.expressions == other.expressions

    def __str__(self) -> str:
        return "ORDER BY " + ", ".join(str(order_expr) for order_expr in self.expressions)


class Limit(BaseClause):
    """The LIMIT clause restricts the number of output rows returned by the database system.

    Each clause can specify an OFFSET (which is probably only meaningful if there is also an ORDER BY clause) and the
    actual LIMIT. Note that some database systems might use a non-standard syntax for such clauses.
    """

    def __init__(self, *, limit: Optional[int] = None, offset: Optional[int] = None) -> None:
        if limit is None and offset is None:
            raise ValueError("LIMIT and OFFSET cannot be both unspecified")
        self._limit = limit
        self._offset = offset

        hash_val = hash((self._limit, self._offset))
        super().__init__(hash_val)

    @property
    def limit(self) -> Optional[int]:
        """Get the maximum number of rows in the result set."""
        return self._limit

    @property
    def offset(self) -> Optional[int]:
        """Get the offset within the result set (i.e. number of first rows to skip)."""
        return self._offset

    def columns(self) -> set[base.ColumnReference]:
        return set()

    def iterexpressions(self) -> Iterable[expr.SqlExpression]:
        return []

    def itercolumns(self) -> Iterable[base.ColumnReference]:
        return []

    __hash__ = BaseClause.__hash__

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.limit == other.limit and self.offset == other.offset

    def __str__(self) -> str:
        limit_str = f"LIMIT {self.limit}" if self.limit is not None else ""
        offset_str = f"OFFSET {self.offset}" if self.offset is not None else ""
        return limit_str + offset_str
