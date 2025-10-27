"""
Implementation of the Cardinality Estimation Benchmark (CEB) algorithm to automatically generate workload queries based on
different templates.

While the original CEB was introduced in [1]_, the allowed template settings and their interaction was barely documented.
Therefore, we provide our own implementation that is (hopefully) better documented and understandable.

Generally speaking you do not need to interact with this module directly. Instead, there exists a high-level CLI tool
called ``ceb-generator.py`` in the ``tools`` directory. It exposes the most relevant options and can be used instead.

References
----------
[1] Parimarjan Negi et al.: "Flow-Loss: Learning Cardinality Estimates That Matter" (PVLDB 2021)
"""

from __future__ import annotations

import collections
import pathlib
import random
import tomllib
from collections.abc import Iterable
from typing import Any, Literal, NewType, Optional

import numpy as np

from ..db import postgres
from ..db._db import Database
from ..qal import formatter, parser
from ..qal._qal import ColumnReference, SqlQuery, TableReference
from ..util._errors import StateError
from ..util.misc import DependencyGraph
from .workloads import Workload

# we introduce a bunch of type aliases to prevent types like dict[str, str]
TemplatedQuery = NewType("TemplatedQuery", str)
"""The raw query text with placeholders for the predicate values. Due to the placeholders, this won't be a valid SQL query."""

ColumnName = NewType("ColumnName", str)
"""Reference to a column with a placeholder predicate."""

PredicateName = NewType("PredicateName", str)
"""A predicate that produces a subset of the placeholder values."""

PredicateType = Literal["=", "<", ">", "<=", ">=", "<>", "LIKE", "ILIKE", "IN"]
"""The operand of a placeholder predicate."""

PlaceholderName = NewType("PlaceholderName", str)
"""The actual placeholder key that will be replace in a `TemplatedQuery`."""

PlaceHolderValue = Any
"""The selected value to bind and replace a placeholder."""


def _tuplelize_value(val: Any) -> Any:
    """Leaves scalar values as they are, but converts lists to tuples. This ensures that we can hash them."""
    if isinstance(val, list):
        return tuple(val)
    return val


def _make_options_list(
    options: list[PlaceHolderValue],
) -> list[tuple[PlaceHolderValue]]:
    """Transforms explitic options list into our standardized format: a list of tuples of individual option values."""
    if not options:
        raise ValueError("Must provide at least one option")
    initial_opt = options[0]
    if isinstance(initial_opt, tuple):
        return options
    return [(opt,) for opt in options]


def _remove_weight_col(
    val: tuple[PlaceHolderValue], col_idx: int
) -> tuple[PlaceHolderValue]:
    """Drops the weight column from a pre-weighted tuple."""
    return tuple([elem for i, elem in enumerate(val) if i != col_idx])


class PredicateGenerator:
    """The predicate generator handles the selection of substittion values for a subset of all placeholders in a template.

    Parameters
    ----------
    name : PredicateName
        Alias of the predicate. This might be referenced as a dependency in other predicates.
    provided_keys : list[PlaceholderName]
        The placeholders for which this predicate calculates substitution values.
    template_type : Literal["sql", "list"]
        The inference strategy for the predicate. If "sql", the values are fetched from the `db_connection` using an actual SQL
        query. If "list", the values must be provided as a static list.
    sampling_method : Literal["uniform", "weighted"]
        How to select the final value from a list of candidates. If "uniform", all values have the same probability. If
        "weighted", values that occur multiple times have a higher higher chance of being selected.
    target_columns : list[ColumnName]
        Columns corresponding to the placeholders. This has to be indexed in the same order as `provided_keys`. Columns must
        be listed explicitly to ensure that values are escaped properly in the final query. For example, text values require
        surrounding quotes, etc.
    pred_type : list[PredicateType]
        Operands corresponding to the columns and placeholders. This has to be indexed in the same order as `provided_keys`.
        Operands must be listed explicitly in order to ensure that values are formatted and prepared correctly for the final
        query. For example, *LIKE* predicates require insertion of wildcard characters, *IN* predicates require surrounding
        parens, etc.

        Notice that *IN* predicates require that the predicate only computes a single placeholder value. Otherwise it is
        unclear how scalar values from one predicate should correlate to values for the *IN* predicate.
    sql_query : Optional[str], optional
        The actual SQL query to compute the selected values. This query must compute the values in the same order as the
        `provided_keys`. It can contain placeholders that are computed by the predicates listed in `dependencies`. This
        parameter is required for ``template_type="sql"`` and ignored otherwise.
    list_allowed_values : Optional[list[PlaceHolderValue]], optional
        The options to choose from to select a placeholder value. This parameter is required for ``template_type="list"`` and
        ignored otherwise.
    in_pred_min_samples : Optional[int], optional
        For *IN* predicates, this designates the minimum number of values that must be included in the final *IN* predicate.
        Set to 1 by default.
    in_pred_max_samples : Optional[int], optional
        For *IN* predicates this designates the maximum number of values that might be included in the final *IN* predicate.
        This defaults to the total number of candidate values. Since this might be quite a lot, it is recommended to set this
        value to something reasonable.
    count_column_idx : Optional[int], optional
        For weighted sampling this denotes the column that contains pre-calculated weights for each candidate value. If this
        is omitted, the candidate values are assumed to contain duplicates and weights are inferred based on the number of
        occurences of each value. The index is relative to the *SELECT* clause of the SQL query or the list of allowed values.
    dependencies : Optional[list[PredicateName]], optional
        Predicates that compute values referenced in this predicate's SQL query.
    max_tries : Optional[int], optional
        The maximum number of attempts to select a valid value for the predicate. It is necessary to re-try if the selected
        value fails some constraints (e.g. the number of allowed values in an *IN* predicate). In such a case, all dependent
        values are re-drawn as well.
    db_connection : Optional[Database], optional
        The database containing the values to sample from.
    """

    def __init__(
        self,
        name: PredicateName,
        *,
        provided_keys: list[PlaceholderName],
        template_type: Literal["sql", "list"],
        sampling_method: Literal["uniform", "weighted"],
        target_columns: list[ColumnName],
        pred_type: list[PredicateType],
        sql_query: Optional[str] = None,
        list_allowed_values: Optional[list[PlaceHolderValue]] = None,
        in_pred_min_samples: int = 1,
        in_pred_max_samples: Optional[int] = None,
        count_column_idx: Optional[int] = None,
        dependencies: Optional[list[PredicateName]] = None,
        max_tries: Optional[int] = None,
        db_connection: Optional[Database] = None,
    ) -> None:
        self.name = name

        if "IN" in pred_type and len(provided_keys) > 1:
            raise ValueError(
                "IN predicates must only compute a single placeholder value"
            )
        self.pred_type = pred_type

        self.dependencies = dependencies

        if not (len(provided_keys) == len(target_columns) == len(pred_type)):
            raise ValueError(
                "The number of provided keys, target columns, and predicate types must match"
            )
        self._key_lookup = dict(((k, i) for i, k in enumerate(provided_keys)))

        self._parent_generator: QueryTemplate | None = None
        self._template_type = template_type
        self._sampling_method = sampling_method
        self._target_columns = target_columns

        if template_type == "sql" and not sql_query:
            raise ValueError(
                f"SQL query must be provided for sql-typed predicate '{name}'"
            )
        if template_type == "list" and not list_allowed_values:
            raise ValueError(
                f"Option values must be provided for list-typed predicate '{name}'"
            )
        self._sql_query = sql_query
        self._list_allowed_values = (
            _make_options_list(list_allowed_values) if list_allowed_values else None
        )

        self._count_col_idx = (
            count_column_idx - 1 if count_column_idx is not None else None
        )

        self._in_pred_min_samples = in_pred_min_samples
        self._in_pred_max_samples = in_pred_max_samples

        self._max_tries = max_tries
        self._db_connection = db_connection

        self._selected_values: list = []

    @property
    def placeholders(self) -> Iterable[PlaceholderName]:
        """Provides all placeholder keys that are computed by this predicate."""
        return self._key_lookup.keys()

    def choose_predicate_values(self) -> None:
        """Draws a valid value for the predicate based on the provided strategy.

        Raises
        ------
        SamplingError
            If no valid value passing all constraints could be found within the maximum number of tries.
        """
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

            if not all(
                self._value_passes_constraints(key, selected_value)
                for key in self._key_lookup.keys()
            ):
                redraw_dependent_values = True
                selected_value = None
                continue
            else:
                break

        if not selected_value:
            raise SamplingError(
                f"Did not find a valid value for predicate '{self.name}'"
            )
        self._selected_values = selected_value

    def fetch_value(self, key: PlaceholderName) -> PlaceHolderValue:
        """Provides the selected value for a specific placeholder key.

        Raises
        ------
        StateError
            If no value has been selected yet via `choose_predicate_values()`.
        """
        self._assert_values_available()
        self._assert_valid_key(key)
        value_idx = self._key_lookup[key]
        return self._selected_values[value_idx]

    def selected_values(self) -> dict[PlaceholderName, PlaceHolderValue]:
        """Provides all selected values for the placeholders computed by this predicate.

        Raises
        ------
        StateError
            If no value has been selected yet via `choose_predicate_values()`.
        """
        self._assert_values_available()
        return {k: self._selected_values[i] for k, i in self._key_lookup.items()}

    def column_for(self, key: PlaceholderName) -> ColumnName:
        """Provides the column that is computed by a specific placeholder key."""
        self._assert_valid_key(key)
        value_idx = self._key_lookup[key]
        return self._target_columns[value_idx]

    def predicate_for(self, key: PlaceholderName) -> PredicateType:
        """Provides the operand of the predicate computed by a specifc placeholder key."""
        self._assert_valid_key(key)
        value_idx = self._key_lookup[key]
        return self.pred_type[value_idx]

    def _next_predicate_value(
        self, redraw_dependent_values: bool
    ) -> tuple[PlaceHolderValue]:
        """Calculates the next (tuple of) placeholder values based on the specified selection strategy.

        This is the main workhorse method which delegates to all further more specialized methods, e.g. to actually draw
        a value from SQL, or to select a value with weighted propabilities, etc.

        Parameters
        ----------
        redraw_dependent_values : bool
            Whether all predicates that supply dependent values to this predicates should select new values. This is important
            if we already tried to select a new value but it did not pass all of our constraints.
        """
        if self._template_type == "list" and self._list_allowed_values is not None:
            candidate_values = self._list_allowed_values
        elif self._template_type == "sql":
            candidate_values = self._collect_candidate_values_from_sql(
                redraw_dependent_values
            )
        else:
            raise ValueError(f"Unknown template type: '{self._template_type}'")

        if self.pred_type == ["IN"]:
            selected_value = self._draw_multi_values(candidate_values)
        else:
            selected_value = self._draw_scalar_value(candidate_values)

        if not isinstance(selected_value, list) and not isinstance(
            selected_value, tuple
        ):
            selected_value = [selected_value]
        return selected_value

    def _collect_candidate_values_from_sql(
        self, redraw_dependent_values: bool
    ) -> list[tuple[PlaceHolderValue]]:
        """Provides all possible candidate values based on an SQL query.

        This method is also responsible for generating an adequate SQL query by subsituting all dependent values.

        Parameters
        ----------
        redraw_dependent_values : bool
            Whether all predicates that supply dependent values to this predicates should select new values. This is important
            if we already tried to select a new value but it did not pass all of our constraints.

        Raises
        ------
        SamplingError
            If the query did not provide any results. This might happen if we have an unlucky selection of dependent values
            and is no big deal since we can simply try again (as controlled by the higher-up methods).
        """
        sql_query = self._sql_query

        for dep in self.dependencies:
            dependent_values = self._parent_generator.selected_values(
                dep, refresh=redraw_dependent_values
            )
            sql_query = self._parent_generator.substitute_placeholders(
                sql_query, dependent_values
            )

        candidate_values = self._db_connection.execute_query(sql_query, raw=True)
        if not candidate_values:
            raise SamplingError(f"No values found for predicate '{self.name}'")
        return [tuple(candidate) for candidate in candidate_values]

    def _draw_scalar_value(
        self, candidate_values: list[tuple[PlaceHolderValue]]
    ) -> tuple[PlaceHolderValue]:
        """Selects a single value from the candidates according to the specified sampling strategy."""
        if self._sampling_method == "uniform":
            # For uniform selection duplicate occurences of the same value should not increase their chance of selection.
            # Therefore we need to make sure that no value is present more than once.
            unique_values = list(set(candidate_values))
            selected_val = random.choice(unique_values)
            return selected_val

        elif self._sampling_method == "weighted":
            # For weighted mode, we can either receive the desired weights along with the candidate values, or we might need
            # to calculate them ourselves.
            # In the first case, each candidate value tuple also contains a weight entry that is designated by the
            # `count_col_idx` attribute.
            # In the latter case, each occurence of the same candidate value counts as a weight increase, hence we can just
            # select one of the values at uniform probability without eliminating duplicates.
            weights: list[int] | None = (
                [val[self._count_col_idx] for val in candidate_values]
                if self._count_col_idx is not None
                else None
            )
            selected_val = random.choices(candidate_values, weights=weights, k=1)[
                0
            ]  # choices always returns a list!

            if self._count_col_idx is not None:
                # for pre-weighted lists our selected value does not only contain the actual data, but also the weight column
                # since the weight column can be located at an arbitrary position and should not become part of the actual
                # "payload", we need to filter the selected value before returning it
                selected_val = _remove_weight_col(selected_val, self._count_col_idx)
            return selected_val

        else:
            raise ValueError(f"Unknown sampling method: '{self._sampling_method}'")

    def _draw_multi_values(
        self, candidate_values: list[tuple[PlaceHolderValue]]
    ) -> tuple[PlaceHolderValue]:
        """Selects placeholder values for *IN* predicates according to the specified sampling strategy."""
        if len(candidate_values[0]) != 1 or (
            self._count_col_idx and len(candidate_values[0]) != 2
        ):
            raise ValueError(
                "IN predicates must only compute a single placeholder value"
            )

        min_values = self._in_pred_min_samples

        if self._sampling_method == "uniform":
            # Uniform sampling is easy: we just need to determine which unique values are available and then choose a
            # correctly-sized subset from them
            candidate_values = list(set(candidate_values))

            max_values = (
                len(candidate_values)
                if self._in_pred_max_samples is None
                else min(self._in_pred_max_samples, len(candidate_values))
            )
            n_values = random.randint(min_values, max_values)

            selected_val = random.sample(candidate_values, k=n_values)
            return [tuple(selected_val)]

        if self._count_col_idx is not None:
            # If weights are already supplied, we just need to extract them
            val_idx = 0 if self._count_col_idx == 1 else 1
            population, weights = zip(
                *[(val[val_idx], val[self._count_col_idx]) for val in candidate_values]
            )
        else:
            # Otherwise we calculate our own weights based on the number of occurences of each value
            counter = collections.Counter([val[0] for val in candidate_values])
            population, weights = zip(*counter.items())

        max_values = (
            len(population)
            if self._in_pred_max_samples is None
            else min(self._in_pred_max_samples, len(population))
        )
        n_values = random.randint(min_values, max_values)

        # We use numpy's random module here because it supports sampling from a population with custom weights as well as
        # without replacement.
        # But numpy expects propabilities instead of weights, so we need calculate them first
        weights = np.array(weights)
        weights = weights / weights.sum()

        rng = np.random.default_rng()
        selected_val: list[PlaceHolderValue] = rng.choice(
            population, size=n_values, p=weights, replace=False
        )
        return [tuple(selected_val)]

    def _value_passes_constraints(
        self, key: PlaceholderName, value: list[PlaceHolderValue]
    ) -> bool:
        """Checks, whether a specific value passes all constraints attached to its placeholder."""
        if self.predicate_for(key) != "IN":
            return True

        max_allowed_values = (
            self._in_pred_max_samples
            if self._in_pred_max_samples is not None
            else len(value)
        )
        return self._in_pred_min_samples <= len(value) <= max_allowed_values

    def _assert_values_available(self) -> None:
        """Raises an error if no values have been selected yet."""
        if not self._selected_values:
            raise StateError(
                "Must first call choose_predicate_values() to select values"
            )

    def _assert_valid_key(self, key: PlaceholderName) -> None:
        """Raises an error if a placeholder is not computed by the current predicate."""
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
    """A query template handles generation of all placeholder values and the replacement process in the final query.

    Parameters
    ----------
    base_query : TemplatedQuery
        The actual query to format. Its placeholders will be replaced by the selected values obtained from the predicate
        generators.
    label : str
        The label of the current query template. Mostly used for debugging purposes.
    table_aliases : dict[str, str]
        A map from alias to physical table name. This is necessary to determine to which table the columns belong without
        parsing the query (which is impossible since the query is not yet valid SQL due to the placeholders). If a table does
        not have an alias, the table name itself can be used as a key. The value has to be empty in this case.
    db_connection : Database
        The database providing the actual candidate values for the placeholders
    """

    def __init__(
        self,
        base_query: TemplatedQuery,
        *,
        label: str,
        table_aliases: dict[str, str],
        db_connection: Database,
    ) -> None:
        self.label = label
        self.base_query = base_query

        # The table aliases should map alias => fully-qualified table name. However, if a table has no alias, its
        # fully-qualified name becomes the key instead. This makes the weird calculation of the target value necessary.
        # The root cause is that a physical table can be referenced with multiple aliases in the same query and other tables
        # might be referenced without any alias, still within the same query. SQL is weird, man!
        self._table_aliases = {
            alias: (tab if tab else alias) for alias, tab in table_aliases.items()
        }

        self._predicate_generators: dict[PredicateName, PredicateGenerator] = {}
        self._generator_lookup: dict[PlaceholderName, PredicateGenerator] = {}

        self._db_conn = db_connection

    @property
    def max_tries(self) -> int:
        """How often a predicate generator may attempt to obtain a valid placeholder value."""
        return 10

    def register_generator(self, generator: PredicateGenerator) -> None:
        """Adds a new predicate generator to the template.

        Raises
        ------
        KeyError
            If another generator has already been registered for one of the provided placeholder keys or the same generator
            name.
        """
        if generator.name in self._predicate_generators:
            raise KeyError(
                f"Predicate '{generator.name}' already registered in template '{self.label}'"
            )

        self._predicate_generators[generator.name] = generator

        for key in generator.placeholders:
            if key in self._generator_lookup:
                raise KeyError(
                    f"Key '{key}' already registered in template '{self.label}'"
                )
            self._generator_lookup[key] = generator

        generator._parent_generator = self

    def selected_values(
        self, predicate: PredicateName, *, refresh: bool = False
    ) -> dict[PlaceholderName, PlaceHolderValue]:
        """Provides the values selected by a specific predicate generator.

        Parameters
        ----------
        predicate : PredicateName
            The generator name.
        refresh : bool, optional
            Whether the generator should re-draw all values. This is useful if some dependent predicate cannot satisfy its
            constraints with the current values.
        """
        if predicate not in self._predicate_generators:
            raise KeyError(
                f"Predicate '{predicate}' not found in template '{self.label}'"
            )

        generator = self._predicate_generators[predicate]
        if refresh:
            generator.choose_predicate_values()
        return generator.selected_values()

    def substitute_placeholders(
        self,
        query: TemplatedQuery,
        selected_values: dict[PlaceholderName, PlaceHolderValue],
    ) -> TemplatedQuery:
        """Replaces all placeholders with their selected values in a specific query.

        The query must not be the `base_query`. For example, it can also be a dependent SQL query of a predicate generator.
        """
        for key, value in selected_values.items():
            generator = self._generator_lookup[key]

            target_column = self._lookup_column(generator.column_for(key))
            column_dtype = self._db_conn.schema().datatype(target_column)
            pred_type = generator.predicate_for(key)

            escaped_placeholder = self._escape_col_value(value, pred_type, column_dtype)
            query = query.replace(f"<<{key}>>", escaped_placeholder)

        return query

    def generate_raw_query(self) -> str:
        """Creates a new SQL query by replacing all placeholders in the base query with appropriate values."""
        dep_graph: DependencyGraph[PredicateGenerator] = DependencyGraph()
        for generator in self._predicate_generators.values():
            dependencies = (
                [self._predicate_generators[dep] for dep in generator.dependencies]
                if generator.dependencies
                else []
            )
            dep_graph.add_task(generator, depends_on=dependencies)

        for generator in dep_graph:
            generator.choose_predicate_values()

        final_query = self.base_query
        for generator in self._predicate_generators.values():
            selected_values = generator.selected_values()
            final_query = self.substitute_placeholders(final_query, selected_values)

        return str(final_query)

    def generate_query(self) -> SqlQuery:
        """Creates a new SQL query by replacing all placeholders in the base query with appropriate values."""
        final_query = self.generate_raw_query()
        return parser.parse_query(final_query)

    def _lookup_column(self, colname: ColumnName) -> ColumnReference:
        """Generates an actual column reference for a specific column name."""
        if "." not in colname:
            tables_without_alias = [
                TableReference(tab)
                for tab, alias in self._table_aliases
                if tab == alias
            ]
            target_table = self._db_conn.schema().lookup_column(
                colname, tables_without_alias
            )
            return ColumnReference(colname, target_table)

        table, column = colname.split(".")
        target_table = self._table_aliases[table]
        table_ref = TableReference(target_table, table)
        return ColumnReference(column, table_ref)

    def _escape_col_value(self, value: PlaceHolderValue, pred_type, dtype: str) -> str:
        """Creates an appropriately escaped string for a placeholder.

        Depending on the predicate type the value might be processed further, e.g. by adding wildcard operators for *LIKE*
        predicates. Likewise, the values might be wrapped by parens for *IN* predicates.
        """
        if isinstance(value, tuple):
            assert pred_type == "IN"
            escaped_values = [self._escape_col_value(v, "=", dtype) for v in value]
            value_text = ", ".join(escaped_values)
            return f"({value_text})"

        if dtype in {"text", "varchar", "char", "character varying"}:
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


def _parse_template_toml(
    path: str | pathlib.Path, db_connection: Database
) -> QueryTemplate:
    """Generates a full query template instance based on its TOML description."""
    contents = {}
    with open(path, "rb") as toml_file:
        contents = tomllib.load(toml_file)

    query_template = QueryTemplate(
        TemplatedQuery(contents["base_sql"]["sql"]),
        label=contents["title"],
        table_aliases=contents["base_sql"]["table_aliases"],
        db_connection=db_connection,
    )

    for raw_predicate in contents["predicates"]:
        parsed_predicate = PredicateGenerator(
            PredicateName(raw_predicate["name"]),
            provided_keys=[
                PlaceholderName(k.removeprefix("<<").removesuffix(">>"))
                for k in raw_predicate["keys"]
            ],
            template_type=raw_predicate["type"],
            sampling_method=raw_predicate["sampling_method"],
            pred_type=raw_predicate["pred_type"],
            target_columns=[ColumnName(c) for c in raw_predicate["columns"]],
            sql_query=raw_predicate.get("sql"),
            list_allowed_values=[
                _tuplelize_value(option) for option in raw_predicate.get("options", [])
            ],
            in_pred_min_samples=raw_predicate.get("min_samples", 1),
            in_pred_max_samples=raw_predicate.get("max_samples"),
            dependencies=[
                PredicateName(d) for d in raw_predicate.get("dependencies", [])
            ],
            db_connection=db_connection,
        )
        query_template.register_generator(parsed_predicate)

    return query_template


def generate_raw_workload(
    path: str | pathlib.Path,
    *,
    queries_per_template: int,
    template_pattern: str = "*.toml",
    db_connection: Optional[Database] = None,
) -> dict[str, str]:
    """Produces an unoptimized workload based on a number of CEB templates.

    In contrast to `generate_workload`, generated queries are not parsed into actual query objects. Instead, the raw query
    text is provided. This function is intended for situations where the parser or the query abstraction cannot yet be used
    to represent the desired structure of the templates.

    Parameters
    ----------
    path : str | pathlib.Path
        The directory containing the template files.
    queries_per_template : int
        The number of queries that should be generated for each template. Queries will be distinguished by increasing label
        numbers.
    template_pattern : str, optional
        A GLOB pattern that all template files must match to be recognized as such. Defaults to *\\*.toml*.
    db_connection : Optional[Database], optional
        The database to use for fetching appropriate candidate values for the placeholders. If omitted, a default Postgres
        connection will be opened.

    Returns
    -------
    dict[str, str]
        The generated workload. Maps query labels to the raw query text.

    Raises
    ------
    SamplingError
        If the sampling algorithm could not satisfy all constraints of its predicates.

    See Also
    --------
    generate_workload
    """
    db_connection = postgres.connect() if db_connection is None else db_connection
    template_dir = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
    if not template_dir.is_dir():
        raise FileNotFoundError(f"Directory '{template_dir}' does not exist")

    templates: list[QueryTemplate] = []
    for template_file in template_dir.glob(template_pattern):
        templates.append(_parse_template_toml(template_file, db_connection))

    max_tries = (
        len(templates) * queries_per_template * 10
    )  # TODO: the user should be able to control this parameter?!
    generated_queries: set[str] = set()
    workload_queries: dict[str, str] = {}
    for template in templates:
        generated_count, num_tries = 0, 0
        while generated_count < queries_per_template and num_tries <= max_tries:
            num_tries += 1
            query = template.generate_raw_query()
            if query in generated_queries:
                if num_tries == max_tries:
                    raise SamplingError(
                        "Could not generate enough unique queries for template {template.label}"
                    )
                continue
            else:
                generated_queries.add(query)
                generated_count += 1

            template_idx = str(
                generated_count
            )  # this works b/c we already incremented the generated_count just above!
            query_label = f"{template.label}-{template_idx}"
            workload_queries[query_label] = query

    return workload_queries


def generate_workload(
    path: str | pathlib.Path,
    *,
    queries_per_template: int,
    name: Optional[str] = None,
    template_pattern: str = "*.toml",
    db_connection: Optional[Database] = None,
) -> Workload[str]:
    """Produces a full workload based on a number of CEB templates.

    Parameters
    ----------
    path : str | pathlib.Path
        The directory containing the template files.
    queries_per_template : int
        The number of queries that should be generated for each template. Queries will be distinguished by increasing label
        numbers.
    name : Optional[str], optional
        The name of the resulting workload.
    template_pattern : str, optional
        A GLOB pattern that all template files must match to be recognized as such. Defaults to *\\*.toml*.
    db_connection : Optional[Database], optional
        The database to use for fetching appropriate candidate values for the placeholders. If omitted, a default Postgres
        connection will be opened.

    Returns
    -------
    Workload[str]
        The generated workload. Queries are differentiated by labels based on the template names.

    Raises
    ------
    SamplingError
        If the sampling algorithm could not satisfy all constraints of its predicates.
    """
    template_dir = path if isinstance(path, pathlib.Path) else pathlib.Path(path)
    raw_workload = generate_raw_workload(
        template_dir,
        queries_per_template=queries_per_template,
        template_pattern=template_pattern,
        db_connection=db_connection,
    )
    workload_queries = {
        label: parser.parse_query(query) for label, query in raw_workload.items()
    }

    return Workload(workload_queries, name=(name if name else ""), root=template_dir)


def persist_workload(
    path: str | pathlib.Path, workload: Workload[str] | dict[str, str]
) -> None:
    """Stores all queries of a workload with one query per file in a specific directory.

    Files are named according to the query lables.
    """
    path = pathlib.Path(path) if isinstance(path, str) else path
    query_iter = (
        workload.entries() if isinstance(workload, Workload) else workload.items()
    )
    query_formatter = (
        formatter.format_quick if isinstance(workload, Workload) else lambda x: x
    )
    for label, query in query_iter:
        query_file = path / f"{label}.sql"
        with open(query_file, "w") as query_file:
            query_file.write(query_formatter(query) + "\n")


class SamplingError(RuntimeError):
    """Error to indicate that something went wrong during the sampling process.

    This error can either be exposed to the user to indicate that something might be wrong with the templates (e.g. constraints
    that are too restrictive or sampling that is too random), or within the sampling process. In the latter case this denotes
    situations that will be resolved automatically within the generation process and without user intervention.
    """

    def __init__(self, message) -> None:
        super().__init__(message)
