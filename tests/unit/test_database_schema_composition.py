"""Tests for the parts of `DatabaseSchema` that are composed from the leaf lookups: ``schema[...]``, `as_graph`,
`join_equivalence_keys`, `join_equivalence_classes` and the alias handling of the multi-column predicates.

The leaf lookups are answered by a `StaticSchema`, so the composition runs for real while the catalog is a
declarative spec (the SQL of the leaves is covered in `tests/unit/test_database_schema.py`). The schema below has one
simple and one composite foreign key and a composite primary key::

    parent(a, b, c)    PRIMARY KEY (a, b), UNIQUE (b, c), INDEX parent_c_idx (c)
    other(id)          PRIMARY KEY (id)
    child(a, b, o, v)  FOREIGN KEY (a, b) REFERENCES parent(a, b), FOREIGN KEY (o) REFERENCES other(id)

Several composite-key blind spots of the composition are pinned instead of asserted, see the `Documents a real bug`
tests at the end: composite foreign keys are invisible to the graph, `TableInfo` and the equivalence classes, the
equivalence classes contain `ForeignKeyRef` objects instead of columns, and unknown keys do not raise `KeyError`.
"""

from __future__ import annotations

import networkx as nx
import pytest

from postbound import ColumnReference, TableReference
from postbound.db import ForeignKeyRef, PrimaryKeyRef
from postbound.db._db import ColumnInfo, TableInfo
from tests.doubles import StaticSchema, column_spec, foreign_key_spec

PARENT = TableReference("parent")
OTHER = TableReference("other")
CHILD = TableReference("child")
CHILD_ALIASED = TableReference("child", "c")

P_A, P_B, P_C = (ColumnReference(name, PARENT) for name in "abc")
O_ID = ColumnReference("id", OTHER)
C_A, C_B, C_O = (ColumnReference(name, CHILD) for name in ("a", "b", "o"))

COMPOSITE_FK = ForeignKeyRef([C_A, C_B], [P_A, P_B])
SIMPLE_FK = ForeignKeyRef([C_O], [O_ID])


@pytest.fixture
def schema() -> StaticSchema:
    return StaticSchema(
        {
            "parent": [column_spec("a", nullable=False), column_spec("b", nullable=False), "c"],
            "other": [column_spec("id", nullable=False, primary_key=True)],
            "child": ["a", "b", column_spec("o", references="other.id"), column_spec("v", "text")],
        },
        primary_keys={"parent": ["a", "b"]},
        foreign_keys=[foreign_key_spec("child", ["a", "b"], "parent", ["a", "b"])],
        unique={"parent": [["b", "c"]]},
        indexes={"parent": {"parent_c_idx": ["c"]}},
    )


# -- schema[column] ----------------------------------------------------------------------------------


def test_column_info_of_a_simple_primary_key_column(schema: StaticSchema) -> None:
    info = schema[O_ID]

    assert info == ColumnInfo(
        column=O_ID, table=OTHER, datatype="integer", nullable=False, primary_key=True, indexed=True, unique=False
    )


def test_column_info_of_a_composite_primary_key_member_is_not_marked_as_primary_key_or_indexed(
    schema: StaticSchema,
) -> None:
    """`ColumnInfo` documents that a composite key is not reported for its individual columns."""
    info = schema[P_A]

    assert info.primary_key is False
    assert info.indexed is False
    assert info.unique is False
    assert info.nullable is False


def test_column_info_of_a_member_of_a_composite_unique_constraint_is_not_unique(schema: StaticSchema) -> None:
    assert schema[P_C].unique is False
    assert schema.is_unique([P_B, P_C]) is True


def test_column_info_reports_a_secondary_index_on_the_column_itself(schema: StaticSchema) -> None:
    assert schema[P_C].indexed is True


def test_column_info_ignores_table_aliases(schema: StaticSchema) -> None:
    info = schema[ColumnReference("o", CHILD_ALIASED)]

    assert info.column == C_O
    assert info.table == CHILD


# -- schema[table] -----------------------------------------------------------------------------------


def test_table_info_of_a_table_with_composite_primary_key(schema: StaticSchema) -> None:
    info = schema[PARENT]

    assert isinstance(info, TableInfo)
    assert info.table == PARENT
    assert info.primary_key == PrimaryKeyRef([P_A, P_B])
    assert [col.column.name for col in info.columns] == ["a", "b", "c"]
    assert info[P_C].indexed is True


def test_table_info_lists_the_indexes_of_the_table(schema: StaticSchema) -> None:
    info = schema[PARENT]

    assert dict(info.indexes) == {
        "parent_pkey": {P_A, P_B},
        "parent_b_c_key": {P_B, P_C},
        "parent_c_idx": {P_C},
    }


def test_table_info_of_a_table_without_primary_key(schema: StaticSchema) -> None:
    assert schema[CHILD].primary_key is None


def test_table_info_reports_simple_foreign_keys_in_both_directions(schema: StaticSchema) -> None:
    assert SIMPLE_FK in schema[CHILD].outgoing_foreign_keys
    assert schema[OTHER].incoming_foreign_keys == {SIMPLE_FK}
    assert schema[OTHER].outgoing_foreign_keys == set()


def test_table_info_is_a_mapping_over_its_columns(schema: StaticSchema) -> None:
    info = schema[CHILD]

    assert len(info) == 4
    assert list(info) == [ColumnReference(name, CHILD) for name in ("a", "b", "o", "v")]
    assert info["v"].datatype == "text"
    assert info[C_O].column == C_O
    with pytest.raises(KeyError, match="not found"):
        info["missing"]


def test_table_info_ignores_table_aliases(schema: StaticSchema) -> None:
    assert schema[CHILD_ALIASED].table == CHILD


def test_string_keys_are_interpreted_as_tables(schema: StaticSchema) -> None:
    assert schema["parent"].table == PARENT


def test_unsupported_key_types_raise_a_key_error(schema: StaticSchema) -> None:
    with pytest.raises(KeyError, match="Unsupported key type"):
        schema[42]  # type: ignore[index]


# -- as_graph ----------------------------------------------------------------------------------------


def test_as_graph_has_a_node_per_table_with_its_primary_key(schema: StaticSchema) -> None:
    graph = schema.as_graph()

    assert set(graph.nodes) == {PARENT, OTHER, CHILD}
    assert graph.nodes[PARENT]["primary_key"] == PrimaryKeyRef([P_A, P_B])
    assert graph.nodes[CHILD]["primary_key"] is None
    assert [col.name for col in graph.nodes[PARENT]["columns"]] == ["a", "b", "c"]
    assert graph.nodes[CHILD]["data_type"][ColumnReference("v", CHILD)] == "text"


def test_as_graph_has_an_edge_from_the_referencing_to_the_referenced_table(schema: StaticSchema) -> None:
    graph = schema.as_graph()

    assert graph.edges[CHILD, OTHER]["foreign_keys"] == [SIMPLE_FK]
    assert not graph.has_edge(OTHER, CHILD)


def test_as_graph_collects_parallel_foreign_keys_on_one_edge() -> None:
    schema = StaticSchema(
        {"r": [column_spec("x", references="s.id"), column_spec("y", references="s.id")], "s": ["id"]}
    )

    graph = schema.as_graph()

    assert len(graph.edges[TableReference("r"), TableReference("s")]["foreign_keys"]) == 2


def test_as_graph_is_directed(schema: StaticSchema) -> None:
    assert isinstance(schema.as_graph(), nx.DiGraph)


# -- join equivalence --------------------------------------------------------------------------------


def test_join_equivalence_of_a_schema_without_foreign_keys_is_empty() -> None:
    schema = StaticSchema({"r": ["a"], "s": ["b"]})

    assert schema.join_equivalence_keys() == {}
    assert list(schema.join_equivalence_classes()) == []


# -- alias handling of the multi-column predicates ---------------------------------------------------


@pytest.mark.parametrize("columns", [[C_A, C_B], [C_B, C_A]], ids=["definition-order", "reversed"])
def test_foreign_keys_on_a_composite_key_ignores_the_column_order(schema: StaticSchema, columns) -> None:
    assert schema.foreign_keys_on(columns) == {COMPOSITE_FK}


def test_foreign_keys_on_the_columns_of_the_other_foreign_key_is_empty(schema: StaticSchema) -> None:
    assert schema.foreign_keys_on([C_A]) == set()
    assert schema.foreign_keys_on([C_A, C_O]) == set()


# -- known bugs --------------------------------------------------------------------------------------


def test_equivalence_classes_ignore_composite_foreign_keys(schema: StaticSchema) -> None:
    """Pins the intended behaviour: `join_equivalence_keys` only follows simple
    foreign keys, so the columns of ``child(a, b) -> parent(a, b)`` end up in no equivalence class.
    """
    classes = schema.join_equivalence_keys()

    assert C_A not in classes
    assert P_A not in classes


# -- regression tests --------------------------------------------------------------------------------


def test_table_info_includes_composite_foreign_keys(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: ``schema[table]`` took its foreign keys from `as_graph`, which
    only asked `foreign_keys_on` for single columns, so composite keys were missing from both directions.
    """
    assert schema[CHILD].outgoing_foreign_keys == {COMPOSITE_FK, SIMPLE_FK}
    assert schema[PARENT].incoming_foreign_keys == {COMPOSITE_FK}


def test_as_graph_has_an_edge_for_a_composite_foreign_key(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: see `test_table_info_includes_composite_foreign_keys`."""
    graph = schema.as_graph()

    assert graph.edges[CHILD, PARENT]["foreign_keys"] == [COMPOSITE_FK]


def test_join_equivalence_keys_map_columns_to_columns(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: the classes used to contain `ForeignKeyRef` objects instead of
    the referenced columns, because the loop still assumed that `foreign_keys_on` returns columns.
    """
    classes = schema.join_equivalence_keys()

    assert classes[C_O] == {C_O, O_ID}
    assert classes[O_ID] == {C_O, O_ID}


def test_join_equivalence_classes_contain_columns(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: see `test_join_equivalence_keys_map_columns_to_columns`."""
    assert list(schema.join_equivalence_classes()) == [{C_O, O_ID}]


def test_schema_lookup_of_an_unknown_table_raises_a_key_error(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: ``schema[table]`` raised `networkx.NetworkXError` for unknown
    tables (the table was missing from the graph), which a `Mapping` must not do.
    """
    with pytest.raises(KeyError, match="not found"):
        schema[TableReference("unknown")]


def test_contains_reports_tables_and_columns_without_raising(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: ``table in schema`` raised instead of answering *False*."""
    assert TableReference("unknown") not in schema
    assert CHILD_ALIASED in schema
    assert C_O in schema
    assert ColumnReference("missing", CHILD) not in schema


@pytest.mark.parametrize(
    ("lookup", "plain", "aliased"),
    [
        (lambda schema, col: schema.foreign_keys_on(col), C_O, ColumnReference("o", CHILD_ALIASED)),
        (lambda schema, col: schema.indexes_on(col), P_C, ColumnReference("c", TableReference("parent", "p"))),
    ],
    ids=["foreign_keys_on", "indexes_on"],
)
def test_composed_lookups_ignore_table_aliases_of_the_given_columns(
    schema: StaticSchema, lookup, plain: ColumnReference, aliased: ColumnReference
) -> None:
    """Regression guard for the composite key rework: `foreign_keys_on` and `indexes_on` compared the stored
    (alias-free) columns with the aliased input columns and therefore never matched.
    """
    assert lookup(schema, plain)
    assert lookup(schema, aliased) == lookup(schema, plain)


def test_contains_accepts_string_keys_for_tables_and_columns(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: `__contains__` answered *False* for every ``str`` key although
    ``schema[key]`` works.
    """
    assert "parent" in schema
    assert "child.o" in schema
    assert "child.missing" not in schema
    assert "nothere.o" not in schema


def test_getitem_resolves_qualified_column_strings(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: ``"table.column"`` was split as ``column, table``."""
    assert schema["child.o"].column == C_O
    with pytest.raises(KeyError, match="not found"):
        schema["child.missing"]
    with pytest.raises(KeyError, match="not found"):
        schema["missing"]


def test_contains_accepts_columns_of_an_aliased_table(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: the column was unaliased, but compared with the columns of the
    *aliased* table, so ``c.o in schema`` (``child AS c``) was false.
    """
    assert ColumnReference("o", CHILD_ALIASED) in schema


def test_columns_of_an_aliased_table_carry_no_alias(schema: StaticSchema) -> None:
    assert list(schema.columns(CHILD_ALIASED)) == list(schema.columns(CHILD))


def test_contains_answers_false_for_unqualified_unknown_names(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: ``"missing" in schema`` split off a column from a single
    component and raised ``ValueError: Could not parse table reference with components []``.
    """
    assert "missing" not in schema
    assert "unknown.schema.table" not in schema


def test_contains_handles_fully_qualified_column_strings(schema: StaticSchema) -> None:
    """Regression guard for the composite key rework: a key with four components (``catalog.schema.table.column``) was
    passed to `TableReference.from_components` as a table, which accepts three at most, so ``in`` raised a `ValueError`
    instead of answering. Keys with more than four components can never be valid and must answer *False*.
    """
    assert "catalog.schema.child.o" in schema
    assert "catalog.schema.child.missing" not in schema
    assert "a.b.c.d" not in schema
    assert "a.b.c.d.e" not in schema
