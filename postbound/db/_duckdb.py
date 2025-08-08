# We name this file _duckdb instead of duckdb to avoid conflicts with the official duckdb package. Do not change this!
# The module is available in the __init__ of the db package under the duckdb name. This solves our problems for now.
from __future__ import annotations

import textwrap
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Optional

from .. import optimizer, qal
from .._core import (
    Cardinality,
    ColumnReference,
    Cost,
    JoinOperator,
    PhysicalOperator,
    ScanOperator,
    TableReference,
)
from .._qep import QueryPlan
from ..optimizer import (
    HintType,
    JoinTree,
    PhysicalOperatorAssignment,
    PlanParameterization,
)
from ..qal import SqlQuery, transform
from ..util import Version, jsondict
from ._db import (
    Cursor,
    Database,
    DatabasePool,
    DatabaseSchema,
    DatabaseStatistics,
    HintService,
    OptimizerInterface,
    QueryCacheWarning,
    UnsupportedDatabaseFeatureError,
)
from .postgres import HintParts, PostgresExplainClause, PostgresLimitClause


def _simplify_result_set(result_set: list[tuple[Any]]) -> Any:
    """Implementation of the result set simplification logic outlined in `Database.execute_query`.

    Parameters
    ----------
    result_set : list[tuple[Any]]
        Result set to simplify: each entry in the list corresponds to one row in the result set and each component of the
        tuples corresponds to one column in the result set

    Returns
    -------
    Any
        The simplified result set: if the result set consists just of a single row, this row is unwrapped from the list. If the
        result set contains just a single column, this is unwrapped from the tuple. Both simplifications are also combined,
        such that a result set of a single row of a single column is turned into the single value.
    """
    # simplify the query result as much as possible: [(42, 24)] becomes (42, 24) and [(1,), (2,)] becomes [1, 2]
    # [(42, 24), (4.2, 2.4)] is left as-is
    if not result_set:
        return []

    result_structure = result_set[0]  # what do the result tuples look like?
    if len(result_structure) == 1:  # do we have just one column?
        result_set = [
            row[0] for row in result_set
        ]  # if it is just one column, unwrap it

    if len(result_set) == 1:  # if it is just one row, unwrap it
        return result_set[0]
    return result_set


class DuckDBInterface(Database):
    def __init__(
        self, db: Path, *, system_name: str = "DuckDB", cache_enabled: bool = False
    ) -> None:
        import duckdb as duck

        super().__init__(system_name=system_name, cache_enabled=cache_enabled)

        if cache_enabled:
            warnings.warn(
                "DuckDB interface does not support result caching (yet)",
                QueryCacheWarning,
            )

        self._dbfile = db

        self._db = duck.connect(db)
        self._cur = self._db.cursor()

        self._schema = DuckDBSchema(self)
        self._optimizer = DuckDBOptimizer(self)
        self._hinting = DuckDBHintService(self)

    def schema(self) -> DuckDBSchema:
        return self._schema

    def statistics(
        self,
        *,
        emulated: bool = False,
        enable_emulation_fallback: bool = True,
        cache_enabled: Optional[bool] = True,
    ) -> DatabaseStatistics:
        return DuckDBStatistics(
            self,
            emulated=emulated,
            enable_emulation_fallback=enable_emulation_fallback,
            cache_enabled=cache_enabled,
        )

    def hinting(self) -> HintService:
        return self._hinting

    def optimizer(self) -> OptimizerInterface:
        return self._optimizer

    def execute_query(
        self,
        query: SqlQuery | str,
        *,
        cache_enabled: Optional[bool] = None,
        raw: bool = False,
    ) -> Any:
        if (
            isinstance(query, SqlQuery)
            and query.hints
            and query.hints.preparatory_statements
        ):
            for preparatory_statement in query.hints.preparatory_statements:
                self._cur.execute(preparatory_statement)
            query = transform.drop_hints(query, preparatory_statements_only=True)
        if isinstance(query, SqlQuery):
            query = self._hinting.format_query(query)

        if cache_enabled:
            warnings.warn(
                "DuckDB interface does not support result caching (yet)",
                QueryCacheWarning,
            )

        self._cur.execute(query)
        raw_result = self._cur.fetchall()
        return raw_result if raw else _simplify_result_set(raw_result)

    def database_name(self) -> str:
        self._cur.execute("SELECT CURRENT_DATABASE();")
        db_name = self._cur.fetchone()[0]
        return db_name

    def database_system_version(self) -> Version:
        self._cur.execute("SELECT version();")
        raw_ver: str = self._cur.fetchone()[0]
        raw_ver = raw_ver.removeprefix("v")
        raw_ver = raw_ver.split("-")[0]  # remove the build information
        return Version(raw_ver)

    def cursor(self) -> Cursor:
        return self._cur

    def close(self) -> None:
        self._cur.close()
        self._db.close()

    def reset_connection(self) -> None:
        import duckdb

        try:
            self.close()
        except Exception:
            pass

        self._db = duckdb.connect(self._dbfile)
        self._cur = self._db.cursor()

    def describe(self) -> jsondict:
        base_info = {
            "system_name": self.system_name(),
            "system_version": self.database_system_version(),
            "database": self.database_name(),
        }

        schema_info: list[jsondict] = []
        for table in self._schema.tables():
            column_info: list[jsondict] = []

            for column in self._schema.columns(table):
                column_info.append(
                    {
                        "column": str(column),
                        "indexed": self._schema.has_index(column),
                        "foreign_keys": self._schema.foreign_keys_on(column),
                    }
                )

            schema_info.append(
                {
                    "table": str(table),
                    "n_rows": self.statistics().total_rows(table, emulated=True),
                    "columns": column_info,
                    "primary_key": self._schema.primary_key_column(table),
                }
            )

        base_info["schema_info"] = schema_info
        return base_info


class DuckDBSchema(DatabaseSchema):
    def __init__(self, db: DuckDBInterface) -> None:
        super().__init__(db, prep_placeholder="?")

    def has_secondary_index(self, column: ColumnReference) -> bool:
        if not column.is_bound():
            raise ValueError(
                f"Cannot check index status for {column}: Column is not bound to a table"
            )

        schema_placeholder = "?" if column.table.schema else "current_schema()"

        query_template = textwrap.dedent(f"""
            SELECT ddbi.index_name
            FROM duckdb_indexes() ddbi
            WHERE ddbi.table_name = ?
                AND ltrim(rtrim(ddbi.expressions, ']'), '[') = ?
                AND ddbi.database_name = current_database()
                AND ddbi.schema_name = {schema_placeholder}
            """)

        params = [column.table.full_name, column.name]
        if column.table.schema:
            params.append(column.table.schema)

        self._db.cursor().execute(query_template, parameters=params)
        result_set = self._db.cursor().fetchone()

        return result_set is not None

    def indexes_on(self, column: ColumnReference) -> set[str]:
        if not column.is_bound():
            raise ValueError(
                f"Cannot retrieve indexes for {column}: Column is not bound to a table"
            )

        schema_placeholder = "?" if column.table.schema else "current_schema()"

        # The query template is much more complicated here, due to the different semantics of the constraint_column_usage
        # view. For UNIQUE constraints, the column is the column that is constrained. However, for foreign keys, the column
        # is the column that is being referenced.
        # Notice that this template is different from the vanilla template provided by the default implementation: in the
        # first part, we query from duckdb_indexes() instead of information_schema.key_column_usage!

        query_template = textwrap.dedent(f"""
            SELECT ddbi.index_name
            FROM duckdb_indexes() ddbi
            WHERE ddbi.table_name = ?
                AND ltrim(rtrim(ddbi.expressions, ']'), '[') = ?
                AND ddbi.database_name = current_database()
                AND ddbi.schema_name = {schema_placeholder}
            UNION
            SELECT tc.constraint_name
            FROM information_schema.table_constraints tc
                JOIN information_schema.constraint_column_usage ccu
                    ON tc.constraint_name = ccu.constraint_name
                    AND tc.table_name = ccu.table_name
                    AND tc.table_schema = ccu.table_schema
                    AND tc.table_catalog = ccu.table_catalog
            WHERE tc.constraint_type IN ('PRIMARY KEY', 'UNIQUE')
                AND ccu.table_name = ?
                AND ccu.column_name = ?
                AND ccu.table_catalog = current_database()
                AND ccu.table_schema = {schema_placeholder}
            """)

        # Due to the UNION query, we need to repeat the placeholders. While the implementation is definitely not elegant,
        # this solution is arguably better than relying on named parameters which might or might not be supported by the
        # target database.
        params = [column.table.full_name, column.name]
        if column.table.schema:
            params.append(column.table.schema)
        params.extend([column.table.full_name, column.name])
        if column.table.schema:
            params.append(column.table.schema)

        self._db.cursor().execute(query_template, params)
        result_set = self._db.cursor().fetchall()

        return {row[0] for row in result_set}


class DuckDBStatistics(DatabaseStatistics):
    def __init__(
        self,
        db: DuckDBInterface,
        *,
        emulated: bool = False,
        enable_emulation_fallback: bool = True,
        cache_enabled: Optional[bool] = True,
    ) -> None:
        super().__init__(
            db,
            emulated=emulated,
            enable_emulation_fallback=enable_emulation_fallback,
            cache_enabled=cache_enabled,
        )

    def _retrieve_total_rows_from_stats(self, table: TableReference) -> Optional[int]:
        schema_placeholder = "?" if table.schema else "current_schema()"

        query_template = textwrap.dedent(f"""
            SELECT estimated_size
            FROM duckdb_tables()
            WHERE table_name = ?
                AND database_name = current_database()
                AND schema_name = {schema_placeholder}
            """)

        params = [table.full_name]
        if table.schema:
            params.append(table.schema)

        self._db.cursor().execute(query_template, parameters=params)
        result_set = self._db.cursor().fetchone()

        if not result_set:
            return None

        return result_set[0] if result_set[0] is not None else None

    def _retrieve_distinct_values_from_stats(
        self, column: ColumnReference
    ) -> Optional[int]:
        raise UnsupportedDatabaseFeatureError(
            self._db, "distinct value count statistics."
        )

    def _retrieve_min_max_values_from_stats(
        self, column: ColumnReference
    ) -> Optional[tuple[Any, Any]]:
        raise UnsupportedDatabaseFeatureError(self._db, "min/max value statistics.")

    def _retrieve_most_common_values_from_stats(
        self, column: ColumnReference, k: int
    ) -> Sequence[tuple[Any, int]]:
        raise UnsupportedDatabaseFeatureError(self._db, "most common value statistics.")


class DuckDBOptimizer(OptimizerInterface):
    def __init__(self, db: DuckDBInterface) -> None:
        self._db = db

    def query_plan(self, query: SqlQuery | str) -> QueryPlan:
        raise NotImplementedError

    def analyze_plan(self, query: SqlQuery) -> QueryPlan:
        raise NotImplementedError

    def cardinality_estimate(self, query: SqlQuery | str) -> Cardinality:
        raise NotImplementedError

    def cost_estimate(self, query: SqlQuery | str) -> Cost:
        raise NotImplementedError


class DuckDBHintService(HintService):
    def __init__(self, db: DuckDBInterface) -> None:
        self._db = db

    def generate_hints(
        self,
        query: SqlQuery,
        plan: Optional[QueryPlan] = None,
        *,
        join_order: Optional[JoinTree] = None,
        physical_operators: Optional[PhysicalOperatorAssignment] = None,
        plan_parameters: Optional[PlanParameterization] = None,
    ) -> SqlQuery:
        adapted_query = query
        if adapted_query.explain and not isinstance(
            adapted_query.explain, PostgresExplainClause
        ):
            adapted_query = qal.transform.replace_clause(
                adapted_query, PostgresExplainClause(adapted_query.explain)
            )
        if adapted_query.limit_clause and not isinstance(
            adapted_query.limit_clause, PostgresLimitClause
        ):
            adapted_query = qal.transform.replace_clause(
                adapted_query, PostgresLimitClause(adapted_query.limit_clause)
            )

        has_partial_hints = any(
            param is not None
            for param in (join_order, physical_operators, plan_parameters)
        )
        if plan is not None and has_partial_hints:
            raise ValueError(
                "Can only hint an entire query plan, or individual parts, not both."
            )

        if plan is not None:
            join_order = optimizer.jointree_from_plan(plan)
            physical_operators = optimizer.operators_from_plan(plan)
            plan_parameters = optimizer.parameters_from_plan(plan)

        if join_order is not None:
            hint_parts = self._generate_join_order_hint(join_order)
        else:
            hint_parts = HintParts.empty()

        hint_parts = hint_parts if hint_parts else HintParts.empty()

        if physical_operators:
            operator_hints = self._generate_operator_hints(physical_operators)
            hint_parts = hint_parts.merge_with(operator_hints)

        if plan_parameters:
            plan_hints = self._generate_parameter_hints(plan_parameters)
            hint_parts = hint_parts.merge_with(plan_hints)

        if hint_parts:
            adapted_query = self._add_hint_block(adapted_query, hint_parts)

        return adapted_query

    def format_query(self, query: SqlQuery) -> str:
        # DuckDB uses the Postgres SQL dialect, so this part is easy..
        if query.explain:
            query = transform.replace_clause(
                query, PostgresExplainClause(query.explain)
            )
        return qal.format_quick(query, flavor="postgres")

    def supports_hint(self, hint: PhysicalOperator | HintType) -> bool:
        return hint in {
            ScanOperator.SequentialScan,
            ScanOperator.IndexScan,
            JoinOperator.NestedLoopJoin,
            JoinOperator.HashJoin,
            JoinOperator.SortMergeJoin,
            HintType.LinearJoinOrder,
            HintType.BushyJoinOrder,
            HintType.Cardinality,
            HintType.Operator,
        }

    def _generate_join_order_hint(self, join_tree: JoinTree) -> HintParts:
        if len(join_tree) < 3:
            # we can't force the join direction anyway, so there's no point in generating a hint if there is just a single join
            return HintParts.empty()

        def recurse(join_tree: JoinTree) -> str:
            if join_tree.is_scan():
                return join_tree.base_table.identifier()

            lhs = recurse(join_tree.outer_child)
            rhs = recurse(join_tree.inner_child)

            return f"({lhs} {rhs})"

        join_order = recurse(join_tree)
        hint_parts = HintParts([], [f"JoinOrder({join_order})"])
        return hint_parts

    def _generate_operator_hints(self, ops: PhysicalOperatorAssignment) -> HintParts:
        if not ops:
            return HintParts.empty()

        hints: list[str] = []
        for tab, scan in ops.scan_operators.items():
            match scan.operator:
                case ScanOperator.SequentialScan:
                    op_txt = "SeqScan"
                case ScanOperator.IndexScan:
                    op_txt = "IdxScan"
                case _:
                    raise UnsupportedDatabaseFeatureError(self._db, scan.operator)
            tab_txt = tab.identifier()
            hints.append(f"{op_txt}({tab_txt})")

        for intermediate, join in ops.join_operators.items():
            match join.operator:
                case JoinOperator.NestedLoopJoin:
                    op_txt = "NestLoop"
                case JoinOperator.HashJoin:
                    op_txt = "HashJoin"
                case JoinOperator.SortMergeJoin:
                    op_txt = "MergeJoin"
                case _:
                    raise UnsupportedDatabaseFeatureError(self._db, join.operator)

            intermediate_txt = self._intermediate_to_hint(intermediate)
            hints.append(f"{op_txt}({intermediate_txt})")

        if ops.intermediate_operators:
            raise UnsupportedDatabaseFeatureError(
                self._db,
                "intermediate operators",
            )

        global_settings: list[str] = []
        for param, val in ops.global_settings.items():
            match param:
                case ScanOperator.SequentialScan:
                    param_txt = "enable_seqscan"
                case ScanOperator.IndexScan:
                    param_txt = "enable_indexscan"
                case JoinOperator.NestedLoopJoin:
                    param_txt = "enable_nestloop"
                case JoinOperator.HashJoin:
                    param_txt = "enable_hashjoin"
                case JoinOperator.SortMergeJoin:
                    param_txt = "enable_mergejoin"
                case _:
                    raise UnsupportedDatabaseFeatureError(self._db, param)

            val_txt = "on" if val else "off"
            global_settings.append(f"{param_txt} = {val_txt}")

        return HintParts(global_settings, hints)

    def _generate_parameter_hints(self, parameters: PlanParameterization) -> HintParts:
        if not parameters:
            return HintParts.empty()

        hints: list[str] = []
        for intermediate, card in parameters.cardinality_hints.items():
            if not card.is_valid():
                continue
            intermediate_txt = self._intermediate_to_hint(intermediate)
            hints.append(f"Card({intermediate_txt} #{card})")

        if parameters.parallel_worker_hints:
            raise UnsupportedDatabaseFeatureError(self._db, "parallel worker hints")

        global_settings: list[str] = []
        for param, val in parameters.system_specific_settings.items():
            if isinstance(val, str):
                val_txt = f"'{val}'"
            else:
                val_txt = str(val)

            global_settings.append(f"{param} = {val_txt};")

        return HintParts(global_settings, hints)

    def _add_hint_block(self, query: SqlQuery, hint_parts: HintParts) -> SqlQuery:
        if not hint_parts:
            return query

        local_hints = ["/*=quack_lab="]
        for local_hint in hint_parts.hints:
            local_hints.append(f"  {local_hint}")
        local_hints.append(" */")

        hints = qal.Hint("\n".join(hint_parts.settings), "\n".join(local_hints))
        return qal.transform.add_clause(query, hints)

    def _intermediate_to_hint(self, intermediate: Iterable[TableReference]) -> str:
        """Convert an iterable of TableReferences to a string representation."""
        return " ".join(table.identifier() for table in intermediate)


def connect(
    db: str | Path,
    *,
    name: str = "duckdb",
    refresh: bool = False,
    private: bool = False,
) -> DuckDBInterface:
    db_pool = DatabasePool.get_instance()
    if name in db_pool and not refresh:
        return db_pool.retrieve_database(name)

    db = Path(db) if isinstance(db, str) else db
    duckdb_instance = DuckDBInterface(Path(db))

    if not private:
        db_pool.register_database(name, duckdb_instance)

    return duckdb_instance
