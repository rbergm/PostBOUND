"""Tests for `PrimaryKeyRef` and `ForeignKeyRef` in `postbound.db._db` -- the key descriptions of a `DatabaseSchema`.

Both are plain value objects (they need no database), so everything is tier 0 and the inputs are built directly from
`ColumnReference`s. They replaced the former single-column `ForeignKeyRef` namedtuple and the column-valued primary
key, and model *composite* keys: the column order is preserved (it mirrors the key definition), while the
comparison helpers (`pk_is`, `fk_is`, `referenced_is`) and equality ignore it.

Two quirks are pinned instead of asserted, see the `Documents a real bug` tests: hashing disagrees with equality for
permuted column orders, and `ForeignKeyRef` equality forgets which foreign key column maps to which referenced column.
"""

from __future__ import annotations

import pytest

from postbound import ColumnReference, TableReference, UnboundColumnError
from postbound.db import ForeignKeyRef, PrimaryKeyRef
from postbound.util import jsonize

R = TableReference("r")
S = TableReference("s")
R_ALIASED = TableReference("r", "x")

R_A = ColumnReference("a", R)
R_B = ColumnReference("b", R)
S_C = ColumnReference("c", S)
S_D = ColumnReference("d", S)


# -- PrimaryKeyRef -----------------------------------------------------------------------------------


def test_primary_key_exposes_table_and_columns_in_definition_order() -> None:
    pk = PrimaryKeyRef([R_B, R_A])

    assert pk.table == R
    assert list(pk.columns) == [R_B, R_A]


def test_primary_key_drops_table_aliases() -> None:
    pk = PrimaryKeyRef([ColumnReference("a", R_ALIASED)])

    assert pk.table == R
    assert pk.columns[0] == R_A
    assert pk.columns[0].table.alias == ""


def test_primary_key_behaves_like_a_sequence_of_columns() -> None:
    pk = PrimaryKeyRef([R_A, R_B])

    assert len(pk) == 2
    assert pk[0] == R_A
    assert pk[1] == R_B
    assert list(pk) == [R_A, R_B]
    assert R_B in pk
    assert S_C not in pk
    assert "b" not in pk


@pytest.mark.parametrize(
    ("columns", "error", "message"),
    [
        ([], ValueError, "cannot be empty"),
        ([R_A, S_C], ValueError, "same table"),
        ([R_A, ColumnReference("b")], UnboundColumnError, "Unbound columns"),
    ],
    ids=["empty", "multiple-tables", "unbound-column"],
)
def test_primary_key_rejects_invalid_columns(
    columns: list[ColumnReference], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        PrimaryKeyRef(columns)


def test_primary_key_accepts_any_iterable() -> None:
    pk = PrimaryKeyRef(col for col in (R_A, R_B))

    assert list(pk) == [R_A, R_B]


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (R_A, False),
        ([R_A], False),
        ([R_A, R_B], True),
        ([R_B, R_A], True),
        ({R_A, R_B}, True),
        ([R_A, R_B, ColumnReference("c", R)], False),
        ([S_C, S_D], False),
    ],
    ids=["single", "single-in-list", "exact", "reordered", "as-set", "superset", "other-table"],
)
def test_primary_key_pk_is_compares_columns_regardless_of_order(candidate, expected: bool) -> None:
    assert PrimaryKeyRef([R_A, R_B]).pk_is(candidate) is expected


def test_primary_key_pk_is_for_a_simple_key() -> None:
    pk = PrimaryKeyRef([R_A])

    assert pk.pk_is(R_A) is True
    assert pk.pk_is([R_A]) is True
    assert pk.pk_is(R_B) is False


def test_primary_key_is_simple_and_simplify() -> None:
    simple, composite = PrimaryKeyRef([R_A]), PrimaryKeyRef([R_A, R_B])

    assert simple.is_simple() is True
    assert simple.simplify() == R_A
    assert composite.is_simple() is False
    with pytest.raises(ValueError, match="composite primary key"):
        composite.simplify()


def test_primary_key_equality_ignores_column_order_but_not_the_table() -> None:
    assert PrimaryKeyRef([R_A, R_B]) == PrimaryKeyRef([R_B, R_A])
    assert PrimaryKeyRef([R_A]) != PrimaryKeyRef([R_A, R_B])
    assert PrimaryKeyRef([R_A]) != PrimaryKeyRef([ColumnReference("a", S)])
    assert PrimaryKeyRef([R_A]) != R_A


def test_primary_key_equal_instances_hash_equal() -> None:
    aliased = PrimaryKeyRef([ColumnReference("a", R_ALIASED), ColumnReference("b", R_ALIASED)])

    assert hash(PrimaryKeyRef([R_A, R_B])) == hash(aliased)


def test_primary_key_is_usable_as_a_set_member() -> None:
    assert len({PrimaryKeyRef([R_A]), PrimaryKeyRef([R_A]), PrimaryKeyRef([R_B])}) == 2


def test_primary_key_str_and_repr_name_the_columns() -> None:
    pk = PrimaryKeyRef([R_A, R_B])

    assert str(pk) == "PRIMARY KEY r(a, b)"
    assert "PrimaryKeyRef" in repr(pk)


def test_primary_key_is_json_serializable() -> None:
    serialized = jsonize.to_json(PrimaryKeyRef([R_A, R_B]))

    assert serialized is not None

    assert '"columns"' in serialized
    assert '"a"' in serialized and '"b"' in serialized


def test_primary_key_is_importable_from_the_public_package() -> None:
    from postbound.db import PrimaryKeyRef as PublicPrimaryKeyRef

    assert PublicPrimaryKeyRef is PrimaryKeyRef


# -- ForeignKeyRef -----------------------------------------------------------------------------------


def test_foreign_key_exposes_both_sides() -> None:
    fk = ForeignKeyRef([R_A, R_B], [S_C, S_D])

    assert fk.fk_table == R
    assert list(fk.fk_cols) == [R_A, R_B]
    assert fk.referenced_table == S
    assert list(fk.referenced_cols) == [S_C, S_D]


def test_foreign_key_drops_table_aliases_on_both_sides() -> None:
    fk = ForeignKeyRef([ColumnReference("a", R_ALIASED)], [ColumnReference("c", TableReference("s", "y"))])

    assert fk.fk_table == R
    assert fk.referenced_table == S
    assert fk.fk_cols[0].table.alias == ""
    assert fk.referenced_cols[0].table.alias == ""


@pytest.mark.parametrize(
    ("fk_cols", "referenced_cols", "error", "message"),
    [
        ([R_A, R_B], [S_C], ValueError, "same length"),
        ([], [], ValueError, "cannot be empty"),
        ([R_A, S_C], [S_C, S_D], ValueError, "Foreign key columns must belong to the same table"),
        ([R_A, R_B], [S_C, R_B], ValueError, "Referenced columns must belong to the same table"),
        ([ColumnReference("a")], [S_C], UnboundColumnError, "foreign key"),
        ([R_A], [ColumnReference("c")], UnboundColumnError, "referenced key"),
    ],
    ids=[
        "length-mismatch",
        "empty",
        "fk-multiple-tables",
        "referenced-multiple-tables",
        "unbound-fk",
        "unbound-target",
    ],
)
def test_foreign_key_rejects_invalid_columns(fk_cols, referenced_cols, error: type[Exception], message: str) -> None:
    with pytest.raises(error, match=message):
        ForeignKeyRef(fk_cols, referenced_cols)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (R_A, False),
        ([R_A, R_B], True),
        ([R_B, R_A], True),
        ([R_A], False),
        ([S_C, S_D], False),
    ],
    ids=["single", "exact", "reordered", "subset", "referenced-side"],
)
def test_foreign_key_fk_is_compares_the_referencing_columns_regardless_of_order(candidate, expected: bool) -> None:
    assert ForeignKeyRef([R_A, R_B], [S_C, S_D]).fk_is(candidate) is expected


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (S_C, False),
        ([S_C, S_D], True),
        ([S_D, S_C], True),
        ([S_C], False),
        ([R_A, R_B], False),
    ],
    ids=["single", "exact", "reordered", "subset", "referencing-side"],
)
def test_foreign_key_referenced_is_compares_the_referenced_columns_regardless_of_order(
    candidate, expected: bool
) -> None:
    assert ForeignKeyRef([R_A, R_B], [S_C, S_D]).referenced_is(candidate) is expected


def test_foreign_key_is_simple_and_simplify() -> None:
    simple, composite = ForeignKeyRef([R_A], [S_C]), ForeignKeyRef([R_A, R_B], [S_C, S_D])

    assert simple.is_simple() is True
    assert simple.simplify_fk_cols() == R_A
    assert simple.simplify_referenced_cols() == S_C
    assert composite.is_simple() is False
    with pytest.raises(ValueError, match="composite foreign key"):
        composite.simplify_fk_cols()
    with pytest.raises(ValueError, match="composite foreign key"):
        composite.simplify_referenced_cols()


@pytest.mark.parametrize(
    "other",
    [
        ForeignKeyRef([R_A], [S_C]),
        ForeignKeyRef([R_A, R_B], [S_C, ColumnReference("e", S)]),
        ForeignKeyRef([R_A, ColumnReference("z", R)], [S_C, S_D]),
        ForeignKeyRef(
            [R_A, R_B], [ColumnReference("c", TableReference("t")), ColumnReference("d", TableReference("t"))]
        ),
    ],
    ids=["fewer-columns", "other-referenced-column", "other-fk-column", "other-referenced-table"],
)
def test_foreign_key_inequality(other: ForeignKeyRef) -> None:
    assert ForeignKeyRef([R_A, R_B], [S_C, S_D]) != other


def test_foreign_key_is_not_equal_to_a_non_foreign_key() -> None:
    assert ForeignKeyRef([R_A], [S_C]) != (R_A, S_C)


def test_foreign_key_str_and_repr_name_both_sides() -> None:
    fk = ForeignKeyRef([R_A, R_B], [S_C, S_D])

    assert str(fk) == "FOREIGN KEY r(a, b) REFERENCES s(c, d)"
    assert "ForeignKeyRef" in repr(fk)


def test_foreign_key_is_json_serializable() -> None:
    serialized = jsonize.to_json(ForeignKeyRef([R_A], [S_C]))

    assert serialized is not None

    assert '"foreign_key_columns"' in serialized
    assert '"referenced_columns"' in serialized


def test_foreign_key_is_importable_from_the_public_package() -> None:
    from postbound.db import ForeignKeyRef as PublicForeignKeyRef

    assert PublicForeignKeyRef is ForeignKeyRef


# -- known bugs --------------------------------------------------------------------------------------


# -- regression tests --------------------------------------------------------------------------------


def test_primary_key_hash_does_not_depend_on_the_column_order() -> None:
    """Regression guard for the composite key rework: `PrimaryKeyRef.__eq__` compares sets of columns, but `__hash__`
    used to hash the column *tuple*, so equal keys with permuted columns hashed differently.
    """
    forward, backward = PrimaryKeyRef([R_A, R_B]), PrimaryKeyRef([R_B, R_A])

    assert forward == backward
    assert hash(forward) == hash(backward)
    assert len({forward, backward}) == 1


def test_foreign_key_equal_instances_hash_equal() -> None:
    """Regression guard for the composite key rework: equal foreign keys must hash equal (aliases included)."""
    aliased = ForeignKeyRef([ColumnReference("a", R_ALIASED)], [ColumnReference("c", TableReference("s", "y"))])

    assert aliased == ForeignKeyRef([R_A], [S_C])
    assert hash(aliased) == hash(ForeignKeyRef([R_A], [S_C]))


def test_foreign_key_equality_distinguishes_which_column_maps_to_which() -> None:
    """Regression guard for the composite key rework: `__eq__` used to compare the two sides as independent sets, so
    ``R(a, b) -> S(c, d)`` equalled ``R(a, b) -> S(d, c)``, although the constraints differ.
    """
    straight = ForeignKeyRef([R_A, R_B], [S_C, S_D])
    crossed = ForeignKeyRef([R_A, R_B], [S_D, S_C])

    assert straight != crossed


def test_foreign_key_equality_ignores_the_order_of_a_consistently_permuted_key() -> None:
    """Regression guard for the composite key rework: equality used to compare the ordered column tuples, so the same
    constraint listed in another column order compared unequal (and hashed differently).
    """
    forward, backward = ForeignKeyRef([R_A, R_B], [S_C, S_D]), ForeignKeyRef([R_B, R_A], [S_D, S_C])

    assert forward == backward
    assert hash(forward) == hash(backward)
    assert len({forward, backward}) == 1


def test_foreign_key_mapping_pairs_each_column_with_its_partner() -> None:
    fk = ForeignKeyRef([R_B, R_A], [S_D, S_C])

    assert dict(fk.mapping()) == {R_B: S_D, R_A: S_C}
    assert list(fk.mapping()) == [R_B, R_A]  # ordered like the foreign key columns


def test_foreign_key_mapping_of_a_simple_key() -> None:
    assert dict(ForeignKeyRef([R_A], [S_C]).mapping()) == {R_A: S_C}


def test_foreign_key_column_info_types_are_importable_from_the_public_package() -> None:
    from postbound.db import ColumnInfo, TableInfo

    assert ColumnInfo.__name__ == "ColumnInfo"
    assert TableInfo.__name__ == "TableInfo"
