
import warnings
from dataclasses import dataclass
from typing import List
from attr import attr

import psycopg2


class TableRef:
    @staticmethod
    def virtual(alias: str) -> "TableRef":
        return TableRef(None, alias, True)

    def __init__(self, full_name: str, alias: str, virtual: bool = False):
        self.full_name = full_name
        self.alias = alias
        self.is_virtual = virtual

    def has_attr(self, attr_name) -> bool:
        if not isinstance(attr_name, str):
            warnings.warn("Treating non-string attribute as false: " + str(attr_name))
            return False

        table_qualifier = self.alias + "."
        return attr_name.startswith(table_qualifier)

    def bind_attribute(self, attr_name) -> str:
        return f"{self.alias}.{attr_name}"

    def to_mosp(self):
        if self.is_virtual:
            raise ValueError("Can not convert virtual tables")
        return {"value": self.full_name, "name": self.alias}

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.is_virtual:
            return f"{self.alias} (virtual)"
        return f"{self.full_name} AS {self.alias}"


@dataclass
class AttributeRef:
    src_table: TableRef
    attribute: str

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"{self.src_table.alias}.{self.attribute}"


class DBSchema:
    def __init__(self, cursor: "psycopg2.cursor"):
        self.cursor = cursor

    def lookup_attribute(self, attribute_name: str, candidate_tables: List[TableRef]):
        for table in [tab for tab in candidate_tables if not tab.is_virtual]:
            columns = self._fetch_columns(table.full_name)
            if attribute_name in columns:
                return table
        raise KeyError(f"Attribute not found: {attribute_name} in candidates {candidate_tables}")

    def _fetch_columns(self, table_name):
        base_query = "SELECT column_name FROM information_schema.columns WHERE table_name = %s"
        self.cursor.execute(base_query, (table_name,))
        result_set = self.cursor.fetchall()
        return [col[0] for col in result_set]
