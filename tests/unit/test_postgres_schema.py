"""Tests for `PostgresSchema` in `postbound.postgres._pg` -- the Postgres-specific part of the schema lookups.

`PostgresSchema` inherits the generic information_schema implementation (covered in
`tests/unit/test_database_schema.py`) and overrides `tables` and `all_indexes_on`, which query the *pg_index* system
catalog because Postgres keeps indexes that are not backed by a constraint. The SQL is driven through a
`ScriptedCursor`; what a real server answers (and whether it accepts the statement) is checked by the live tests.

The `PostgresSchema` constructor only needs ``database_name()`` from its database, so a `FakeDatabase` stands in.
"""

from __future__ import annotations

from typing import cast

from postbound import ColumnReference, TableReference
from postbound.db import ForeignKeyRef
from postbound.postgres import PostgresDatabase, PostgresSchema
from tests.doubles import FakeDatabase, ScriptedCursor

TITLE = TableReference("title")

# (index name, schema, table, column, position in the index) -- the shape of the query result
INDEX_ROWS = [
    ("title_pkey", "public", "title", "id", 1),
    ("title_kind_year_idx", "public", "title", "kind_id", 1),
    ("title_kind_year_idx", "public", "title", "production_year", 2),
]


def schema_over(cursor: ScriptedCursor) -> PostgresSchema:
    return PostgresSchema(cast(PostgresDatabase, FakeDatabase(cursor_double=cursor)))


def test_postgres_schema_uses_the_postgres_defaults() -> None:
    schema = schema_over(ScriptedCursor())

    assert schema._prep_placeholder == "%s"
    assert schema._default_schema == "public"
    assert schema._default_catalog == "fake"


def test_all_indexes_on_queries_the_catalog_for_the_table_and_schema() -> None:
    cursor = ScriptedCursor({"pg_index": []})

    schema_over(cursor).all_indexes_on(TITLE)

    call = cursor.calls[0]
    assert call.mentions("pg_index", "pg_attribute", "indkey")
    assert call.parameters == ("title",)
    assert call.mentions("current_schema()")


def test_all_indexes_on_uses_an_explicit_schema_and_ignores_aliases() -> None:
    cursor = ScriptedCursor({"pg_index": []})

    schema_over(cursor).all_indexes_on(TableReference("title", "t", schema="imdb"))

    assert cursor.calls[0].parameters == ("title", "imdb")


def test_all_indexes_on_is_empty_without_indexes() -> None:
    cursor = ScriptedCursor({"pg_index": []})

    assert dict(schema_over(cursor).all_indexes_on(TITLE)) == {}


# -- all_foreign_keys_on ------------------------------------------------------------------------------

FK_FRAGMENT = "contype = 'f'"


def test_all_foreign_keys_on_queries_the_catalog_with_the_unnested_key_arrays() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: []})

    schema_over(cursor).all_foreign_keys_on(TITLE)

    call = cursor.calls[0]
    assert call.mentions("pg_constraint", "unnest(con.conkey, con.confkey) with ordinality")
    assert call.parameters == ("title",)
    assert call.mentions("current_schema()")


# -- all_indexes_on (continued) ----------------------------------------------------------------------


def test_all_indexes_on_binds_an_explicit_schema_after_the_table() -> None:
    """Regression guard for the composite key rework: the schema defaulted to the literal ``"public"`` instead of the
    session's current schema.
    """
    cursor = ScriptedCursor({"pg_index": []})

    schema_over(cursor).all_indexes_on(TableReference("title", schema="imdb"))

    call = cursor.calls[0]
    assert call.parameters == ("title", "imdb")
    assert not call.mentions("current_schema()")


def test_all_indexes_on_reads_index_name_and_column_from_its_result_rows() -> None:
    """Regression guard for the composite key rework: the loop unpacked five-column rows into two names."""
    rows = [("title_pkey", "id"), ("title_kind_year_idx", "kind_id"), ("title_kind_year_idx", "production_year")]
    cursor = ScriptedCursor({"pg_index": rows})

    indexes = schema_over(cursor).all_indexes_on(TITLE)

    assert dict(indexes) == {
        "title_pkey": {ColumnReference("id", TITLE)},
        "title_kind_year_idx": {ColumnReference("kind_id", TITLE), ColumnReference("production_year", TITLE)},
    }


# -- regression tests --------------------------------------------------------------------------------

# (constraint, referencing column, target schema, target table, target column) -- the shape of the catalog query
FK_ROWS = [
    ("fk_movie", "movie_id", "public", "title", "id"),
    ("fk_kind", "kind_id", "imdb", "kind", "id"),
    ("fk_kind", "info_type", "imdb", "kind", "type"),
]
MOVIE_INFO = TableReference("movie_info")


def test_all_foreign_keys_on_selects_only_aliases_that_the_query_defines() -> None:
    """Regression guard for the composite key rework: after renaming the aliases of the pg_constraint query, the SELECT
    list still named ``a``, ``t``, ``ra`` and ``tn``, so PostgreSQL rejected every call (``missing FROM-clause entry
    for table "a"``).
    """
    cursor = ScriptedCursor({FK_FRAGMENT: []})

    schema_over(cursor).all_foreign_keys_on(TITLE)

    sql = cursor.calls[0].normalized
    assert sql.startswith("select con.conname, att.attname, tar_nsp.nspname, tar.relname, tar_att.attname")
    for alias in ("att", "tar", "tar_att", "tar_nsp"):
        assert f" {alias} on " in sql or f" {alias} " in sql


def test_all_foreign_keys_on_interprets_its_five_column_result_rows() -> None:
    """Regression guard for the composite key rework: the loop unpacked six values from the five-column rows."""
    cursor = ScriptedCursor({FK_FRAGMENT: FK_ROWS})

    fks = schema_over(cursor).all_foreign_keys_on(MOVIE_INFO)

    kind, title = TableReference("kind", schema="imdb"), TableReference("title")
    assert fks == {
        ForeignKeyRef([ColumnReference("movie_id", MOVIE_INFO)], [ColumnReference("id", title)]),
        ForeignKeyRef(
            [ColumnReference("kind_id", MOVIE_INFO), ColumnReference("info_type", MOVIE_INFO)],
            [ColumnReference("id", kind), ColumnReference("type", kind)],
        ),
    }


def test_all_foreign_keys_on_keeps_the_pairing_of_a_composite_key() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: [("fk", "b", "public", "p", "b"), ("fk", "a", "public", "p", "a")]})

    (fk,) = schema_over(cursor).all_foreign_keys_on(TableReference("ch"))

    p = TableReference("p")
    assert [col.name for col in fk.fk_cols] == ["b", "a"]
    assert dict(fk.mapping()) == {
        ColumnReference("b", TableReference("ch")): ColumnReference("b", p),
        ColumnReference("a", TableReference("ch")): ColumnReference("a", p),
    }


def test_all_foreign_keys_on_omits_the_default_schema_of_the_target() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: [("fk", "movie_id", "public", "title", "id")]})

    (fk,) = schema_over(cursor).all_foreign_keys_on(MOVIE_INFO)

    assert fk.referenced_table.schema == ""


def test_all_foreign_keys_on_binds_a_schema_after_the_table_name() -> None:
    """Regression guard for the composite key rework: the parameters were assembled as ``(schema, table, schema)``
    although the template has exactly the placeholders table and schema, in this order.
    """
    cursor = ScriptedCursor({FK_FRAGMENT: []})

    schema_over(cursor).all_foreign_keys_on(TableReference("title", schema="imdb"))

    call = cursor.calls[0]
    assert call.parameters == ("title", "imdb")
    assert call.normalized.count("%s") == 2
    assert not call.mentions("current_schema()")


def test_all_foreign_keys_on_ignores_table_aliases_and_accepts_names() -> None:
    cursor = ScriptedCursor({FK_FRAGMENT: []})
    schema = schema_over(cursor)

    schema.all_foreign_keys_on("title")
    schema.all_foreign_keys_on(TableReference("title", "t"))

    assert [call.parameters for call in cursor.calls] == [("title",), ("title",)]
