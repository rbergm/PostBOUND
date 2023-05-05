"""Contains the MySQL implementation of the Database interface."""
from __future__ import annotations

import configparser
import dataclasses
import json
import os
import textwrap
from collections.abc import Sequence
from typing import Any, Optional

import mysql.connector

from postbound.db import db
from postbound.qal import qal, base, clauses, transform
from postbound.util import misc


@dataclasses.dataclass(frozen=True)
class MysqlConnectionArguments:
    user: str
    database: str
    password: str = ""
    host: str = "127.0.0.1"
    port: int = 3306
    use_unicode: bool = True
    charset: str = "utf8mb4"
    autocommit: bool = True
    sql_mode: str = "ANSI"

    def parameters(self) -> dict[str, str | int | bool]:
        return dataclasses.asdict(self)


class MysqlInterface(db.Database):
    def __init__(self, connection_args: MysqlConnectionArguments, system_name: str = "MySQL", *,
                 cache_enabled: bool = True) -> None:
        self.connection_args = connection_args
        self._cnx = mysql.connector.connect(**connection_args.parameters())
        self._cur = self._cnx.cursor(buffered=True)

        self._db_schema = MysqlSchemaInterface(self)
        self._db_stats = MysqlStatisticsInterface(self)
        super().__init__(system_name, cache_enabled=cache_enabled)

    def schema(self) -> db.DatabaseSchema:
        return self._db_schema

    def statistics(self, emulated: bool | None = None, cache_enabled: Optional[bool] = None) -> db.DatabaseStatistics:
        if emulated is not None:
            self._db_stats.emulated = emulated
        if cache_enabled is not None:
            self._db_stats.cache_enabled = cache_enabled
        return self._db_stats

    def execute_query(self, query: qal.SqlQuery | str, *, cache_enabled: Optional[bool] = None) -> Any:
        cache_enabled = cache_enabled or (cache_enabled is None and self._cache_enabled)
        query = self._prepare_query_execution(query)

        if cache_enabled and query in self._query_cache:
            query_result = self._query_cache[query]
        else:
            self._cur.execute(query)
            query_result = self._cur.fetchall()
            if cache_enabled:
                self._query_cache[query] = query_result

        # simplify the query result as much as possible: [(42, 24)] becomes (42, 24) and [(1,), (2,)] becomes [1, 2]
        # [(42, 24), (4.2, 2.4)] is left as-is
        if not query_result:
            return []
        result_structure = query_result[0]  # what do the result tuples look like?
        if len(result_structure) == 1:  # do we have just one column?
            query_result = [row[0] for row in query_result]  # if it is just one column, unwrap it
        return query_result if len(query_result) > 1 else query_result[0]  # if it is just one row, unwrap it

    def cardinality_estimate(self, query: qal.SqlQuery | str) -> int:
        query = self._prepare_query_execution(query, drop_explain=True)
        query_plan = self._obtain_query_plan(query)["query_block"]

        if "nested_loop" in query_plan:
            final_join = query_plan["nested_loop"][-1]
        else:
            final_join = query_plan

        return final_join["table"]["rows_produced_per_join"]

    def cost_estimate(self, query: qal.SqlQuery | str) -> float:
        query = self._prepare_query_execution(query, drop_explain=True)
        query_plan = self._obtain_query_plan(query)
        return query_plan["query_block"]["cost_info"]["query_cost"]

    def database_name(self) -> str:
        self._cur.execute("SELECT DATABASE();")
        db_name = self._cur.fetchone()[0]
        return db_name

    def database_system_version(self) -> misc.Version:
        self._cur.execute("SELECT VERSION();")
        version = self._cur.fetchone()[0]
        return misc.Version(version)

    def inspect(self) -> dict:
        base_info = {
            "system_name": self.database_system_name(),
            "system_version": self.database_system_version(),
            "database": self.database_name(),
            "statistics_settings": {
                "emulated": self._db_stats.emulated,
                "cache_enabled": self._db_stats.cache_enabled
            }
        }
        self._cur.execute("SHOW VARIABLES")
        system_config = self._cur.fetchall()
        base_info["system_settings"] = dict(system_config)
        return base_info

    def reset_connection(self) -> None:
        self._cur.close()
        self._cnx.cmd_reset_connection()
        self._cur = self._cnx.cursor()

    def cursor(self) -> db.Cursor:
        return self._cur

    def close(self) -> None:
        self._cur.close()
        self._cnx.close()

    def _prepare_query_execution(self, query: qal.SqlQuery | str, *, drop_explain: bool = False) -> str:
        """Provides the query in a unified format, taking care of preparatory statements as necessary.

        `drop_explain` can be used to remove any EXPLAIN clauses from the query. Note that all actions that require
        the "semantics" of the query to be known (e.g. EXPLAIN modifications or query hints) and are therefore only
        executed for instances of the qal queries.
        """
        if not isinstance(query, qal.SqlQuery):
            return query

        if drop_explain:
            query = transform.drop_clause(query, clauses.Explain)
        if query.hints and query.hints.preparatory_statements:
            self._cur.execute(query.hints.preparatory_statements)
            query = transform.drop_hints(query, preparatory_statements_only=True)
        return str(query)

    def _obtain_query_plan(self, query: str) -> dict:
        if not query.startswith("EXPLAIN FORMAT = JSON"):
            query = "EXPLAIN FORMAT = JSON " + query
        self._cur.execute(query)
        result = self._cur.fetchone()[0]
        return json.loads(result)


class MysqlSchemaInterface(db.DatabaseSchema):
    def __init__(self, mysql_db: MysqlInterface):
        super().__init__(mysql_db)

    def lookup_column(self, column: base.ColumnReference | str,
                      candidate_tables: list[base.TableReference]) -> base.TableReference:
        column = column.name if isinstance(column, base.ColumnReference) else column
        for table in candidate_tables:
            table_columns = self._fetch_columns(table)
            if column in table_columns:
                return table
        candidate_tables = [tab.full_name for tab in candidate_tables]
        raise ValueError(f"Column {column} not found in candidate tables {candidate_tables}")

    def is_primary_key(self, column: base.ColumnReference) -> bool:
        if not column.table:
            raise base.UnboundColumnError(column)
        index_map = self._fetch_indexes(column.table)
        return index_map.get(column.name, False)

    def has_secondary_index(self, column: base.ColumnReference) -> bool:
        if not column.table:
            raise base.UnboundColumnError(column)
        index_map = self._fetch_indexes(column.table)

        # The index map contains an entry for each attribute that actually has an index. The value is True, if the
        # attribute (which is known to be indexed), is even the Primary Key
        # Our method should return False in two cases: 1) the attribute is not indexed at all; and 2) the attribute
        # actually is the Primary key. Therefore, by assuming it is the PK in case of absence, we get the correct
        # value.
        return not index_map.get(column.name, True)

    def datatype(self, column: base.ColumnReference) -> str:
        if not column.table:
            raise base.UnboundColumnError(column)
        query_template = "SELECT column_type FROM information_schema.columns WHERE table_name = %s AND column_name = %s"
        self._db.cursor().execute(query_template, (column.table.full_name, column.name))
        result_set = self._db.cursor().fetchone()
        return str(result_set[0])

    def _fetch_columns(self, table: base.TableReference) -> list[str]:
        query_template = "SELECT column_name FROM information_schema.columns WHERE table_name = %s"
        self._db.cursor().execute(query_template, (table.full_name,))
        result_set = self._db.cursor().fetchall()
        return [col[0] for col in result_set]

    def _fetch_indexes(self, table: base.TableReference) -> dict[str, bool]:
        index_query = textwrap.dedent("""
            SELECT column_name, column_key = 'PRI'
            FROM information_schema.columns
            WHERE table_name = %s AND column_key <> ''
        """)
        self._db.cursor().execute(index_query, table.full_name)
        result_set = self._db.cursor().fetchall()
        index_map = dict(result_set)
        return index_map


class MysqlStatisticsInterface(db.DatabaseStatistics):
    def __init__(self, mysql_db: MysqlInterface):
        super().__init__(mysql_db)

    def _retrieve_total_rows_from_stats(self, table: base.TableReference) -> Optional[int]:
        count_query = "SELECT table_rows FROM information_schema.tables WHERE table_name = %s"
        self._db.cursor().execute(count_query, table.full_name)
        count = self._db.cursor().fetchone()[0]
        return count

    def _retrieve_distinct_values_from_stats(self, column: base.ColumnReference) -> Optional[int]:
        stats_query = "SELECT cardinality FROM information_schema.statistics WHERE table_name = %s AND column_name = %s"
        self._db.cursor().execute(stats_query, (column.table.full_name, column.name))
        distinct_vals: Optional[int] = self._db.cursor().fetchone()
        if distinct_vals is None and not self.enable_emulation_fallback:
            return distinct_vals
        elif distinct_vals is None:
            return self._calculate_distinct_values(column, cache_enabled=True)
        else:
            return distinct_vals

    def _retrieve_min_max_values_from_stats(self, column: base.ColumnReference) -> Optional[tuple[Any, Any]]:
        if not self.enable_emulation_fallback:
            raise db.UnsupportedDatabaseFeatureError(self._db, "min/max value statistics")
        return self._calculate_min_max_values(column, cache_enabled=True)

    def _retrieve_most_common_values_from_stats(self, column: base.ColumnReference,
                                                k: int) -> Sequence[tuple[Any, int]]:
        if not self.enable_emulation_fallback:
            raise db.UnsupportedDatabaseFeatureError(self._db, "most common values statistics")
        return self._calculate_most_common_values(column, k=k, cache_enabled=True)


def _parse_mysql_connection(config_file: str) -> MysqlConnectionArguments:
    config = configparser.ConfigParser()
    config.read(config_file)
    if not "MYSQL" in config:
        raise ValueError("Malformed MySQL config file: no [MYSQL] section found.")
    mysql_config = config["MYSQL"]

    if "User" not in mysql_config or "Database" not in mysql_config:
        raise ValueError("Malformed MySQL config file: 'User' and 'Database' keys are required in the [MYSQL] section.")
    user = mysql_config["User"]
    database = mysql_config["Database"]

    optional_settings = {}
    for key in ["Password", "Host", "Port", "UseUnicode", "Charset", "AutoCommit", "SqlMode"]:
        if key not in mysql_config:
            continue
        optional_settings[misc.camel_case2snake_case(key)] = mysql_config[key]
    return MysqlConnectionArguments(user, database, **optional_settings)


def connect(*, name: str = "mysql", connection_args: Optional[MysqlConnectionArguments] = None,
            config_file: str = ".mysql_connection.config",
            cache_enabled: Optional[bool] = None, private: bool = False) -> MysqlInterface:
    db_pool = db.DatabasePool.get_instance()
    if config_file and not connection_args:
        if not os.path.exists(config_file):
            raise ValueError("Config file was given, but does not exist: " + config_file)
        connection_args = _parse_mysql_connection(config_file)
    elif not connection_args:
        raise ValueError("Connect string or config file are required to connect to Postgres")

    mysql_db = MysqlInterface(connection_args, system_name=name, cache_enabled=cache_enabled)
    if not private:
        db_pool.register_database(name, mysql_db)
    return mysql_db
