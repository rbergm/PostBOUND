"""
Implementation of the Cardinality Estimation Benchmark (CEB) algorithm to automatically generate workload queries based on
different templates.

References
----------
[1] Parimarjan Negi et al.: "Flow-Loss: Learning Cardinality Estimates That Matter" (PVLDB 2021)
"""

from __future__ import annotations

import pathlib
import random
from collections.abc import Iterable
from typing import Any, Literal, NewType, Optional

import tomli

from .workloads import Workload
from ..db import postgres
from ..db.db import Database
from ..qal import parser, formatter
from ..qal.base import ColumnReference, TableReference
from ..qal.qal import SqlQuery
from ..util.errors import StateError
from ..util.misc import DependencyGraph


TemplatedQuery = NewType("TemplatedQuery", str)
ColumnName = NewType("ColumnName", str)
PredicateName = NewType("PredicateName", str)
PredicateType = Literal["=", "<", ">", "<=", ">=", "<>", "LIKE", "ILIKE", "IN"]
PlaceholderName = NewType("PlaceholderName", str)
PlaceHolderValue = Any


class PredicateGenerator:

    def __init__(self, name: PredicateName,
                 *,
                 provided_keys: list[PlaceholderName],
                 template_type: Literal["sql", "list"],
                 sampling_method: Literal["uniform", "weighted"],
                 pred_type: PredicateType,
                 target_columns: list[ColumnName],

                 sql_query: Optional[str] = None,
                 list_allowed_values: Optional[list[PlaceHolderValue]] = None,

                 in_pred_min_samples: Optional[int] = 1,
                 in_pred_max_samples: Optional[int] = None,

                 dependencies: Optional[list[PredicateName]] = None,
                 max_tries: Optional[int] = None,
                 db_connection: Optional[Database] = None) -> None:
        self.name = name
        self.pred_type = pred_type
        self.dependencies = dependencies

        self._key_lookup = dict(((k, i) for i, k in enumerate(provided_keys)))

        self._parent_generator: QueryTemplate | None = None
        self._template_type = template_type
        self._sampling_method = sampling_method
        self._target_columns = target_columns

        self._sql_query = sql_query
        self._list_allowed_values = list_allowed_values

        self._in_pred_min_samples = in_pred_min_samples
        self._in_pred_max_samples = in_pred_max_samples

        self._max_tries = max_tries
        self._db_connection = db_connection

        self._selected_values: list = []

    @property
    def placeholders(self) -> Iterable[PlaceholderName]:
        return self._key_lookup.keys()

    def choose_predicate_values(self) -> None:
        if self._max_tries is None:
            self._max_tries = self._parent_generator.max_tries

        selected_value: list[PlaceHolderValue] | None = None
        current_try = 0
        redraw_dependent_values = False
        while current_try < self._max_tries:
            current_try += 1
            try:
                selected_value = self._next_predicate_value(redraw_dependent_values)
            except SamplingError:
                # if we did not find any value, retry but make sure to also refresh all dependent values.
                redraw_dependent_values = True
                selected_value = None
                continue

            if not self._value_passes_constraints(selected_value):
                redraw_dependent_values = True
                selected_value = None
                continue
            else:
                break

        if not selected_value:
            raise SamplingError(f"Did not find a valid value for predicate '{self.name}'")
        self._selected_values = selected_value

    def fetch_value(self, key: PlaceholderName) -> PlaceHolderValue:
        self._assert_values_available()
        self._assert_valid_key(key)
        value_idx = self._key_lookup[key]
        return self._selected_values[value_idx]

    def selected_values(self) -> dict[PlaceholderName, PlaceHolderValue]:
        self._assert_values_available()
        return {k: self._selected_values[i] for k, i in self._key_lookup.items()}

    def column_for(self, key: PlaceholderName) -> ColumnName:
        self._assert_valid_key(key)
        value_idx = self._key_lookup[key]
        return self._target_columns[value_idx]

    def _next_predicate_value(self, redraw_dependent_values: bool) -> list[PlaceHolderValue]:
        if self._template_type == "list" and self._list_allowed_values is not None:
            candidate_values = self._list_allowed_values
        elif self._template_type == "sql":
            candidate_values = self._collect_candidate_values_from_sql(redraw_dependent_values)
        else:
            raise ValueError(f"Unknown template type: '{self._template_type}'")

        if self._sampling_method == "uniform":
            unique_values = list(set(candidate_values))
            selected_value = random.choice(unique_values)
        elif self._sampling_method == "weighted":
            selected_value = random.choice(candidate_values)
        else:
            raise ValueError(f"Unknown sampling method: '{self._sampling_method}'")

        if not isinstance(selected_value, list) and not isinstance(selected_value, tuple):
            selected_value = [selected_value]
        return selected_value

    def _collect_candidate_values_from_sql(self, redraw_dependent_values: bool) -> list[PlaceHolderValue]:
        sql_query = self._sql_query

        for dep in self.dependencies:
            dependent_values = self._parent_generator.selected_values(dep, refresh=redraw_dependent_values)
            sql_query = self._parent_generator.substitute_placeholders(sql_query, dependent_values)

        candidate_values = self._db_connection.execute_query(sql_query, raw=True)
        if not candidate_values:
            raise SamplingError(f"No values found for predicate '{self.name}'")
        return candidate_values

    def _value_passes_constraints(self, value: list[PlaceHolderValue]) -> bool:
        if self.pred_type != "IN":
            return True

        max_allowed_values = self._in_pred_max_samples if self._in_pred_max_samples is not None else len(value)
        return self._in_pred_min_samples <= len(value) <= max_allowed_values

    def _assert_values_available(self) -> None:
        if not self._selected_values:
            raise StateError("Must first call choose_predicate_values() to select values")

    def _assert_valid_key(self, key: PlaceholderName) -> None:
        if key not in self._key_lookup:
            raise KeyError(f"Key '{key}' is not provided by template '{self.name}'")

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, type(self)) and value.name == self.name

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.dependencies:
            deps_str = ", ".join(self.dependencies)
            return f"{self.name}({deps_str})"
        return self.name


class QueryTemplate:

    def __init__(self, base_query: TemplatedQuery, *, label: str, table_aliases: dict[str, str],
                 db_connection: Database) -> None:
        self.label = label
        self.base_query = base_query

        # The table aliases should map alias => fully-qualified table name. However, if a table has no alias, its
        # fully-qualified name becomes the key instead. This makes the weird calculation of the target value necessary.
        # The root cause is that a physical table can be referenced with multiple aliases in the same query and other tables
        # might be referenced without any alias, still within the same query. SQL is weird, man!
        self._table_aliases = {alias: (tab if tab else alias) for alias, tab in table_aliases.items()}

        self._predicate_generators: dict[PredicateName, PredicateGenerator] = {}
        self._generator_lookup: dict[PlaceholderName, PredicateGenerator] = {}

        self._db_conn = db_connection

    @property
    def max_tries(self) -> int:
        return 10

    def register_generator(self, generator: PredicateGenerator) -> None:
        if generator.name in self._predicate_generators:
            raise KeyError(f"Predicate '{generator.name}' already registered in template '{self.label}'")

        self._predicate_generators[generator.name] = generator

        for key in generator.placeholders:
            if key in self._generator_lookup:
                raise KeyError(f"Key '{key}' already registered in template '{self.label}'")
            self._generator_lookup[key] = generator

        generator._parent_generator = self

    def selected_values(self, predicate: PredicateName, *, refresh: bool = False) -> dict[PlaceholderName, PlaceHolderValue]:
        if predicate not in self._predicate_generators:
            raise KeyError(f"Predicate '{predicate}' not found in template '{self.label}'")

        generator = self._predicate_generators[predicate]
        if refresh:
            generator.choose_predicate_values()
        return generator.selected_values()

    def substitute_placeholders(self, query: TemplatedQuery,
                                selected_values: dict[PlaceholderName, PlaceHolderValue]) -> TemplatedQuery:
        for key, value in selected_values.items():
            generator = self._generator_lookup[key]

            target_column = self._lookup_column(generator.column_for(key))
            column_dtype = self._db_conn.schema().datatype(target_column)
            pred_type = generator.pred_type

            escaped_placeholder = self._escape_col_value(value, pred_type, column_dtype)
            query = query.replace(f"<<{key}>>", escaped_placeholder)

        return query

    def generate_query(self) -> SqlQuery:
        dep_graph: DependencyGraph[PredicateGenerator] = DependencyGraph()
        for generator in self._predicate_generators.values():
            dependencies = ([self._predicate_generators[dep] for dep in generator.dependencies]
                            if generator.dependencies else [])
            dep_graph.add_task(generator, depends_on=dependencies)

        for generator in dep_graph:
            generator.choose_predicate_values()

        final_query = self.base_query
        for generator in self._predicate_generators.values():
            selected_values = generator.selected_values()
            final_query = self.substitute_placeholders(final_query, selected_values)

        return parser.parse_query(final_query)

    def _lookup_column(self, colname: ColumnName) -> ColumnReference:
        if "." not in colname:
            tables_without_alias = [TableReference(tab) for tab, alias in self._table_aliases if tab == alias]
            target_table = self._db_conn.schema().lookup_column(colname, tables_without_alias)
            return ColumnReference(colname, target_table)

        table, column = colname.split(".")
        target_table = self._table_aliases[table]
        table_ref = TableReference(target_table, table)
        return ColumnReference(column, table_ref)

    def _escape_col_value(self, value: PlaceHolderValue, pred_type, dtype: str) -> str | list[str]:
        if isinstance(value, list):
            assert pred_type == "IN"
            escaped_values = [self._escape_col_value(v, dtype) for v in value]
            value_text = ", ".join(escaped_values)
            return f"({value_text})"

        if dtype in {"text", "varchar", "char"}:
            if pred_type == "LIKE" or pred_type == "ILIKE":
                value = f"'%{value}%'"
            return f"'{value}'"

        return str(value)

    def __hash__(self) -> int:
        return hash(self.label)

    def __eq__(self, value: object) -> bool:
        return isinstance(value, type(self)) and value.label == self.label

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        predicates_str = ", ".join(self._predicate_generators.keys())
        return f"{self.label}({predicates_str})"


def _parse_template_toml(path: str | pathlib.Path, db_connection: Database) -> QueryTemplate:
    contents = {}
    with open(path, "rb") as toml_file:
        contents = tomli.load(toml_file)

    query_template = QueryTemplate(
        TemplatedQuery(contents["base_sql"]["sql"]),
        label=contents["title"],
        table_aliases=contents["base_sql"]["table_aliases"],
        db_connection=db_connection
    )

    for raw_predicate in contents["predicates"]:
        parsed_predicate = PredicateGenerator(
            PredicateName(raw_predicate["name"]),
            provided_keys=[PlaceholderName(k.removeprefix("<<").removesuffix(">>")) for k in raw_predicate["keys"]],
            template_type=raw_predicate["type"],
            sampling_method=raw_predicate["sampling_method"],
            pred_type=raw_predicate["pred_type"],
            target_columns=[ColumnName(c) for c in raw_predicate["columns"]],

            sql_query=raw_predicate.get("sql"),
            list_allowed_values=raw_predicate.get("options"),
            in_pred_min_samples=raw_predicate.get("min_samples"),
            in_pred_max_samples=raw_predicate.get("max_samples"),
            dependencies=[PredicateName(d) for d in raw_predicate.get("dependencies", [])],

            db_connection=db_connection
        )
        query_template.register_generator(parsed_predicate)

    return query_template


def generate_workload(path: str | pathlib.Path, *, queries_per_template: int, name: Optional[str] = None,
                      template_pattern: str = "*.toml", db_connection: Optional[Database] = None) -> Workload[str]:
    db_connection = postgres.connect() if db_connection is None else db_connection
    template_dir = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Directory '{template_dir}' does not exist")

    templates: list[QueryTemplate] = []
    for template_file in template_dir.glob(template_pattern):
        templates.append(_parse_template_toml(template_file, db_connection))

    max_tries = len(templates) * queries_per_template * 10  # TODO: the user should be able to control this parameter?!
    generated_queries: set[SqlQuery] = set()
    workload_queries: dict[str, SqlQuery] = {}
    for template in templates:
        generated_count, num_tries = 0, 0
        while generated_count < queries_per_template and num_tries <= max_tries:
            num_tries += 1
            query = template.generate_query()
            if query in generated_queries:
                if num_tries == max_tries:
                    raise SamplingError("Could not generate enough unique queries for template {template.label}")
                continue
            else:
                generated_queries.add(query)
                generated_count += 1

            template_idx = str(generated_count)  # this works b/c we already incremented the generated_count just above!
            query_label = f"{template.label}-{template_idx}"
            workload_queries[query_label] = query

    return Workload(workload_queries, name=(name if name else ""), root=template_dir)


def persist_workload(path: str | pathlib.Path, workload: Workload[str]) -> None:
    path = pathlib.Path(path) if isinstance(path, str) else path
    for label, query in workload.entries():
        query_file = path / f"{label}.sql"
        with open(query_file, "w") as query_file:
            query_file.write(formatter.format_quick(query))


class SamplingError(RuntimeError):
    def __init__(self, message) -> None:
        super().__init__(message)
