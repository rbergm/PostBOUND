
import abc
from typing import Any, List, Union

from postbound.qal import base, qal

class Database(abc.ABC):
    def __init__(self) -> None:
        self._cache_enabled = False

    @abc.abstractmethod
    def schema(self) -> "DatabaseSchema":
        raise NotImplementedError

    @abc.abstractmethod
    def statistics(self) -> "DatabaseStatistics":
        raise NotImplementedError

    @abc.abstractmethod
    def execute_query(self, query: qal.SqlQuery | str) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def reset_connection(self) -> None:
        raise NotImplementedError

    def _get_cache_enabled(self) -> bool:
        return self._cache_enabled

    def _set_cache_enabled(self, enabled: bool) -> None:
        self._cache_enabled = enabled

    cache_enabled = property(_get_cache_enabled, _set_cache_enabled)


class DatabaseSchema(abc.ABC):
    def __init__(self, db: "Database"):
        self._db = db

    @abc.abstractmethod
    def lookup_column(self, column: base.ColumnReference,
                      candidate_tables: List[base.TableReference]) -> base.TableReference:
        raise NotImplementedError

    @abc.abstractmethod
    def is_primary_key(self, column: base.ColumnReference) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def has_secondary_index(self, column: base.ColumnReference) -> bool:
        raise NotImplementedError

    def has_index(self, column: base.ColumnReference) -> bool:
        return self.is_primary_key(column) or self.has_secondary_index(column)



class DatabaseStatistics(abc.ABC):
    def __init__(self, db: "Database"):
        self._db = db
        self._emulated = True

    def total_rows(self, table: base.TableReference) -> int:
        if self._emulated:
            query_template = "SELECT COUNT(*) FROM {tab}"
            count_query = query_template.format(tab=table.full_name)
            return self._db.execute_query(count_query)
        else:
            return self._retrieve_total_rows_from_stats(table)

    def distinct_values(self, column: base.ColumnReference) -> int:
        if not column.table:
            raise base.UnboundColumnError(column)
        if self._emulated:
            query_template = "SELECT COUNT(DISTINCT {col}) FROM {tab}"
            count_query = query_template.format(col=column.name, tab=column.table.full_name)
            return self._db.execute_query(count_query)
        else:
            return self._retrieve_distinct_values_from_stats(column)

    def most_common_values(self, column: base.ColumnReference, *, k: int = 10) -> list:
        if not column.table:
            raise base.UnboundColumnError(column)
        if self._emulated:
            query_template = "SELECT {col}, COUNT(*) AS n FROM {tab} GROUP BY {col} ORDER BY n DESC, {col} LIMIT {k}"
            count_query = query_template.format(col=column.name, tab=column.table.full_name, k=k)
            return self._db.execute_query(count_query)
        else:
            return self._retrieve_most_common_values_from_stats(column, k)

    @abc.abstractmethod
    def _retrieve_total_rows_from_stats(self, table: base.TableReference) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_distinct_values_from_stats(self, column: base.ColumnReference) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def _retrieve_most_common_values_from_stats(self, column: base.ColumnReference, k: int) -> list:
        raise NotImplementedError

    def _get_emulated(self) -> bool:
        return self._emulated

    def _set_emulated(self, enabled: bool) -> None:
        self._emulated = enabled

    emulated = property(_get_emulated, _set_emulated)


__DB_POOL: Union["DatabasePool", None] = None

class DatabasePool:
    @staticmethod
    def get_instance():
        global __DB_POOL
        pass

    def __init__(self):
        self._pool = {}

    def current_database(self) -> Database:
        pass

    def register_database(self, key: str, db: Database) -> None:
        pass

    def retrieve_database(self, key: str) -> Database:
        pass
