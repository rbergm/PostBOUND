
from __future__ import annotations
import json
import collections
import concurrent
import concurrent.futures
import os
import textwrap
import threading
import re
from typing import Any


import mysql.connector


from postbound.db import db
from postbound.qal import qal, base, transform
from postbound.util import logging, misc as utils


class mysqlInterface(db.Database):
    
    def __init__(self, connect_string: dict[str, Any],system_name: str = "MYSQL",  *, cache_enabled: bool = True) -> None:
       
        self.connect_string = connect_string
        
        self._connection = mysql.connector.connect(**connect_string)
        self._connection.autocommit = True
        self._cursor = self._connection.cursor()
       
        self._db_schema = mysqlSchemaInterface(self)
        self._db_stats = mysqlStatisticsInterface(self)

        super().__init__(system_name, cache_enabled=cache_enabled)

    def schema(self) -> db.DatabaseSchema:
        return self._db_schema
    
    def statistics(self, emulated: bool | None = None, cache_enabled: bool | None = None) -> db.DatabaseStatistics:
        if emulated is not None:
            self._db_stats.emulated = emulated
        if cache_enabled is not None:
            self._db_stats.cache_enabled = cache_enabled
        return self._db_stats
        

    def execute_query(self, query: qal.SqlQuery | str, *, cache_enabled: bool | None = None) -> Any:
        cache_enabled = cache_enabled or (cache_enabled is None and self._cache_enabled)

        if isinstance(query, qal.SqlQuery):
            if query.hints and query.hints.preparatory_statements:
                self._cursor.execute(query.hints.preparatory_statements)
            query = transform.drop_hints(query, preparatory_statements_only=True)
            query = str(query)

        if cache_enabled and query in self._query_cache:
            query_result = self._query_cache[query]
        else:
            self._cursor.execute(query)
            query_result = self._cursor.fetchall()
            if cache_enabled:
                self._query_cache[query] = query_result

        
        if not query_result:
            return []
        result_structure = query_result[0]  
        if len(result_structure) == 1:  
            query_result = [row[0] for row in query_result]  
        return query_result if len(query_result) > 1 else query_result[0]
    
    def cardinality_estimate(self, query: qal.SqlQuery | str) -> int:
        query = str(query)
        if not query.upper().startswith("EXPLAIN FORMAT=JSON"):
            query = "EXPLAIN FORMAT=JSON " + query
        self._cursor.execute(query)
        query_plan_json = self._cursor.fetchone()[0]
        query_plan = json.loads(query_plan_json)
        query_block = query_plan.get("query_block")
        if query_block and "table" in query_block:
            estimate = int(query_block["table"]["rows_examined_per_scan"])
        else:
            estimate = 1
        return estimate
    
    
   
    
    def database_name(self) -> None:
        self._cursor.execute("SELECT DATABASE();")
        db_name = self._cursor.fetchone()[0]
        return db_name
        
    
    def database_system_version(self) -> utils.Version:
        self._cursor.execute("SELECT VERSION();")
        mysql_ver = self._cursor.fetchone()[0]
        # version looks like "8.0.32"
        return utils.Version(str(mysql_ver))

        
            

    def reset_connection(self) -> None:
        self._cursor.close()
        self._connection.rollback()
        self._cursor = self._connection.cursor()

    def cursor(self) -> db.Cursor:
        return self._cursor

    def close(self) -> None:
        self._cursor.close()
        self._connection.close()


class mysqlSchemaInterface(db.DatabaseSchema):
    """Schema-specific parts of the general MYSQL interface."""

    def __init__(self, mysql_db: mysqlInterface) -> None:
        super().__init__(mysql_db)

    def lookup_column(self, column: base.ColumnReference,
                      candidate_tables: list[base.TableReference]) -> base.TableReference:
        for table in candidate_tables:
            table_columns = self._fetch_columns(table)
            if column.name in table_columns:
                return table
        candidate_tables = [table.full_name for table in candidate_tables]
        raise ValueError("Column '{}' not found in candidate tables {}".format(column.name, candidate_tables))
    
    def is_primary_key(self, column: base.ColumnReference) -> bool:
        if not column.table:
            raise base.UnboundColumnError(column)
        index_map = self._fetch_indexes(column.table)
        return index_map.get(column.name, False)
    
    def has_secondary_index(self, column: base.ColumnReference) -> bool:
        if not column.table:
            raise base.UnboundColumnError(column)
        index_map = self._fetch_indexes(column.table)
        return not index_map.get(column.name, True)
    
    def datatype(self, column: base.ColumnReference) -> str:
        if not column.table:
            raise base.UnboundColumnError(column)
        query_template = textwrap.dedent("""
                        SELECT data_type FROM information_schema.columns
                        WHERE BINARY table_name = '{tab}' AND BINARY column_name = '{col}'""")
        query_template = query_template.strip()
        datatype_query = query_template.format(tab=column.table.full_name, col=column.name)
        self._db.cursor().execute(datatype_query)
        result_set = self._db.cursor().fetchone()
        return result_set[0]
    
    def _fetch_indexes(self, table: base.TableReference) -> dict[str, bool]:

        index_query = textwrap.dedent(f""" SELECT COLUMN_NAME AS attname,  CASE WHEN COLUMN_KEY = 'PRI' THEN 'true' ELSE 'false' END AS indisprimary 
                                           FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table.full_name}' 
                                           AND (COLUMN_NAME IN (SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.STATISTICS 
                                           WHERE  TABLE_NAME = '{table.full_name}'
                                           AND INDEX_NAME = 'PRIMARY') OR COLUMN_KEY = 'MUL'); """)
        index_query = index_query.strip()
        
        cursor = self._db.cursor()
        cursor.execute(index_query)
        result_set = cursor.fetchall()
        index_map = {}
        for row in result_set:
            col_name, has_secondary_index = row
            index_map[col_name] = has_secondary_index == 'true'
        return index_map
    
    def _fetch_columns(self, table: base.TableReference) -> list[str]:
        """Retrieves all physical columns for a given table from the MYSQL metadata catalogs."""
        query_template = "SELECT column_name FROM information_schema.columns WHERE BINARY table_name = %s"
        self._db.cursor().execute(query_template, (table.full_name,))
        result_set = self._db.cursor().fetchall()
        return [col[0] for col in result_set]
    

 

class mysqlStatisticsInterface(db.DatabaseStatistics):
    """Statistics-specific parts of the MYSQL interface."""

    def __init__(self, mysql_db: mysqlInterface) -> None:
        super().__init__(mysql_db)    
    
    def _retrieve_total_rows_from_stats(self, table: base.TableReference) -> int:
        count_query = f"SELECT TABLE_ROWS  FROM INFORMATION_SCHEMA.TABLES WHERE BINARY TABLE_NAME = '{table.full_name}'"
        self._db.cursor().execute(count_query)
        count = self._db.cursor().fetchone()[0]
        return count
    
    def _retrieve_distinct_values_from_stats(self, column: base.ColumnReference) -> int:
        dist_query = "SELECT DISTINCTROW CARDINALITY FROM information_schema.statistics WHERE BINARY table_name = %s AND BINARY column_name = %s"
        self._db.cursor().execute(dist_query, (column.table.full_name, column.name))
        dist_values = self._db.cursor().fetchone()[0]
        return dist_values
        
    
    def _retrieve_min_max_values_from_stats(self, column: base.ColumnReference) -> tuple:
        if not self.enable_emulation_fallback:
            raise db.UnsupportedDatabaseFeatureError(self._db, "min/max value statistics")
        return self._calculate_min_max_values(column, cache_enabled=True)
    
    def _retrieve_most_common_values_from_stats(self, column: base.ColumnReference, k: int) -> list:
        return self._calculate_most_common_values(column, k)
    

    

    
    
    
def connect(*, name: str = "mysql", connect_string: str | None = None, config_file: str | None = ".mysql_connection", cache_enabled: bool = True, private: bool = False) -> mysqlInterface:
   
    db_pool = db.DatabasePool.get_instance()
    if config_file and not connect_string:
        if not os.path.exists(config_file):
            raise ValueError("Config file was given, but does not exist: " + config_file)
        with open(config_file, "r") as f:
            connect_string = f.readline().strip()
    elif not connect_string:
        raise ValueError("Connect string or config file are required to connect to MYSQL")

    mysql_db = mysqlInterface(connect_string, system_name=name, cache_enabled=cache_enabled)
    if not private:
        db_pool.register_database(name, mysql_db)
    
    return mysql_db
    
