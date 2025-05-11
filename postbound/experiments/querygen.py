"""Simple randomized query generator."""
from __future__ import annotations

import random
from collections.abc import Generator
from typing import Optional

import networkx as nx

from .. import util
from ..db import DatabasePool, Database, postgres
from ..qal import (
    SqlQuery,
    LogicalOperator, CompoundOperator,
    AbstractPredicate, CompoundPredicate,
    Select, ImplicitFromClause, Where,
    build_query, as_predicate
)
from ..util import networkx as nx_utils
from .._core import TableReference, ColumnReference


def _generate_join_predicates(tables: list[TableReference], *, schema: nx.DiGraph) -> AbstractPredicate:
    """Generates equi-join predicates for specific tables.

    Between each pair of tables, a join predicate is generated if there is a foreign key relationship between the two tables.

    Parameters
    ----------
    tables : list[TableReference]
        The tables to join.
    schema : nx.DiGraph
        Graph for the entire schema. It must contain node for all `tables`, but can contain more nodes.

    Returns
    -------
    AbstractPredicate
        A compound predicate that represents all foreign key joins between the tables. Notice that if there are partitions in
        the schema, the predicate will not span all partitions, leading to cross products between some tables.

    Raises
    ------
    ValueError
        If no foreign key edges are found between the tables.
    """
    predicates: list[AbstractPredicate] = []

    for i, outer_tab in enumerate(tables):

        # Make sure to check each pair of tables only once
        for inner_tab in tables[i + 1:]:

            # Since we are working with a directed graph, we need to check for both FK reference "directions"
            fk_edge = schema.get_edge_data(outer_tab, inner_tab)
            if not fk_edge:
                fk_edge = schema.get_edge_data(inner_tab, outer_tab)
            if not fk_edge:
                continue

            source_col, target_col = fk_edge["fk_col"], fk_edge["referenced_col"]
            join_predicate = as_predicate(source_col, LogicalOperator.Equal, target_col)
            predicates.append(join_predicate)

    if not predicates:
        raise ValueError(f"Found no suitable edges in the schema graph for tables {tables}.")
    return CompoundPredicate.create_and(predicates)


def _generate_filter(column: ColumnReference, *, target_db: Database) -> Optional[AbstractPredicate]:
    """Generates a random filter predicate for a specific column.

    The predicate will be a simple binary predicate using one of the following operators: equality, inequality, greater than,
    or less than. The comparison value is randomly selected from the set of distinct values for the column.

    Parameters
    ----------
    column : ColumnReference
        The column to filter.
    target_db : Database
        The database that contains all allowed values for the column.

    Returns
    -------
    Optional[AbstractPredicate]
        A random filter predicate on the column. If no candidate values are found in the database, *None* is returned.

    """

    # TODO: for text columns we should also generate LIKE predicates
    candidate_operators = [LogicalOperator.Equal, LogicalOperator.NotEqual, LogicalOperator.Greater, LogicalOperator.Less]

    # We need to compute the unique values for the column
    # For Postgres, we can use the TABLESAMPLE clause to reduce the number of rows to materialize by a large number.
    # For all other databases, we need to compute the entire result set.
    # In either case, we should never cache the results of this query, even if this might be way more efficient for continuous
    # sampling of queries. The reason is that the number of distinct values can be huge and we don't want to overload the cache
    estimated_n_rows = target_db.statistics().total_rows(column.table, emulated=False)
    if isinstance(target_db, postgres.PostgresInterface) and estimated_n_rows > 1000:
        distinct_template = """SELECT DISTINCT {col} FROM {tab} TABLESAMPLE BERNOULLI(1)"""
    else:
        distinct_template = """SELECT DISTINCT {col} FROM {tab}"""

    candidate_values = target_db.execute_query(distinct_template.format(col=column.name, tab=column.table.full_name),
                                               cache_enabled=False)
    if not candidate_values:
        return None

    selected_value = random.choice(candidate_values)
    selected_operator = random.choice(candidate_operators)
    filter_predicate = as_predicate(column, selected_operator, selected_value)
    return filter_predicate


def _is_numeric(data_type: str) -> bool:
    """Checks, whether a data type is numeric."""
    return (data_type in {"integer", "smallint", "bigint", "date", "double precision", "real", "numeric", "decimal"}
            or data_type.startswith("time"))


def generate_query(target_db: Optional[Database], *,
                   count_star: bool = False, ignore_tables: Optional[set[TableReference]] = None,
                   min_tables: Optional[int] = None, max_tables: Optional[int] = None,
                   min_filters: Optional[int] = None, max_filters: Optional[int] = None,
                   filter_key_columns: bool = True, numeric_filters: bool = False) -> Generator[SqlQuery, None, None]:
    """A simple randomized query generator.

    The generator selects a random subset of (connected) tables from the schema graph of the target database and builds a
    random number of random filter predicates on the columns of the selected tables.

    The generator will yield new queries until the user stops requesting them, there is no termination condition.

    Parameters
    ----------
    target_db : Optional[Database]
        The database from which queries should be generated. The database is important for two main use-cases:

        1. The schema graph is used to select a (connected) subset of tables
        2. The column values are used to generate filter predicates

        If no database is provided, the current database from the database pool is used.

    count_star : bool, optional
        Whether the resulting queries should contain a *COUNT(\\*)* instead of a plain * *SELECT* clause
    ignore_tables : Optional[set[TableReference]], optional
        An optional set of tables that should never be contained in the generated queries. For Postgres databases, internal
        *pg_XXX* tables are ignored automatically.
    min_tables : Optional[int], optional
        The minimum number of tables that should be contained in each query. Default is 1.
    max_tables : Optional[int], optional
        The maximum number of tables that should be contained in each query. Default is the number of tables in the schema
        graph (minus the ignored tables).
    min_filters : Optional[int], optional
        The minimum number of filter predicates that should be contained in each query. Default is 0.
    max_filters : Optional[int], optional
        The maximum number of filter predicates that should be contained in each query. By default, each column from the
        selected tables can be filtered.
    filter_key_columns : bool, optional
        Whether primary key/foreign key columns should be considered for filtering. This is enabled by default.
    numeric_filters : bool, optional
        Whether only numeric columns should be considered for filtering (i.e. integer, float or time columns). This is disabled
        by default.

    Yields
    ------
    Generator[SqlQuery, None, None]
        A random SQL query

    Examples
    --------
    >>> qgen = generate_query(some_database)
    >>> next(qgen)
    """

    target_db = target_db or DatabasePool.get_instance().current_database()
    db_schema = target_db.schema()

    #
    # Our sampling algorithm is acutally pretty straightforward:
    #
    # 1. We select a random number of tables
    # 2. We select a random number of columns to filter from the selected tables
    # 3. We generate the join predicates between the tables
    # 4. We generate random filter predicates for the selected columns
    # 5. We build the query
    #
    # The hardest part (and the part that takes up the most LOCs), is making sure that we always select from the correct
    # subset.
    #

    schema_graph = db_schema.as_graph()
    if ignore_tables:
        nodes_to_remove = [node for node in schema_graph.nodes if node in ignore_tables]
        schema_graph.remove_nodes_from(nodes_to_remove)
    if isinstance(target_db, postgres.PostgresInterface):
        nodes_to_remove = [node for node in schema_graph.nodes if node.full_name.startswith("pg_")]
        schema_graph.remove_nodes_from(nodes_to_remove)

    min_tables = min_tables or 1
    if not min_tables:
        raise ValueError("min_tables must be at least 1")
    max_tables = max_tables or len(schema_graph.nodes)
    max_tables = min(max_tables, len(schema_graph.nodes))
    if max_tables < min_tables:
        raise ValueError(f"max_tables must be at least as large as min_tables. Got {max_tables} (max) and {min_tables} (min).")

    filter_columns = util.flatten(cols for __, cols in schema_graph.nodes(data="columns"))
    min_filters = min_filters or 0
    max_filters = max_filters or len(filter_columns)

    select_clause = Select.count_star() if count_star else Select.star()

    # We generate new queries until the user asks us to stop.
    while True:
        n_tables = random.randint(min_tables, max_tables)

        # We ensure that we always generate a connected join graph by performing a random walk through the schema graph.
        # This way, we can terminate the walk at any point if we have visited enough tables.
        table_walk = nx_utils.nx_random_walk(schema_graph.to_undirected())
        joined_tables: list[TableReference] = [next(table_walk) for _ in range(n_tables)]

        available_columns: list[ColumnReference] = util.flatten([cols for tab, cols in schema_graph.nodes(data="columns")
                                                                 if tab in set(joined_tables)])
        if not filter_key_columns:
            available_columns = [col for col in available_columns
                                 if not db_schema.is_primary_key(col) and not db_schema.foreign_keys_on(col)]
        if numeric_filters:
            available_columns = [col for col in available_columns
                                 if _is_numeric(schema_graph.nodes[col.table]["data_type"][col])]

        from_clause = ImplicitFromClause.create_for(joined_tables)
        join_predicates = _generate_join_predicates(joined_tables, schema=schema_graph) if n_tables > 1 else None

        current_max_filters = min(max_filters, len(available_columns))
        if current_max_filters <= min_filters:
            # too few columns available, let's just try again
            continue
        else:
            n_filters = random.randint(min_filters, current_max_filters)

        if n_filters > 0 and available_columns:
            cols_to_filter = random.sample(available_columns, n_filters)
            individual_filters = [_generate_filter(col, target_db=target_db) for col in cols_to_filter]
            individual_filters = [pred for pred in individual_filters if pred is not None]
            filter_predicates = CompoundPredicate.create_and(individual_filters) if individual_filters else None
        else:
            filter_predicates = None

        # This is just a bit of optimization to avoid useless nesting inside the WHERE clause
        where_parts: list[AbstractPredicate] = []
        for predicates in (join_predicates, filter_predicates):
            match predicates:
                case None:
                    pass
                case CompoundPredicate(op, children) if op == CompoundOperator.And:
                    where_parts.extend(children)
                case _:
                    where_parts.append(join_predicates)

        where_clause = Where(CompoundPredicate.create_and(where_parts)) if where_parts else None

        query = build_query([select_clause, from_clause, where_clause])
        yield query
