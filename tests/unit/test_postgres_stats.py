"""Tests for `PostgresStatistics` in `postbound.postgres._pg` -- statistics read from the native Postgres catalogs.

`PostgresStatistics` only talks to its database through ``self._db.cursor()``, so a `ScriptedCursor` (wrapped in a
`FakeDatabase`) is a complete seam: the tests run the real catalog queries and the real interpretation of their rows.

The scripted rows are verbatim catalog contents of the Stats database (see the comments above the fixtures), since
the interesting parts of this class are the conventions of *pg_stats*: how `n_distinct` is encoded, which rows the
MCV frequencies are relative to, and how the histogram bounds relate to the MCVs. Those conventions were checked
against all integer columns of Stats and JOB: ANALYZE removes the MCVs before choosing the histogram bounds, so an MCV
is never a bound itself, but it can fall into any bucket or lie below/above the entire histogram.

Whether the catalog queries are accepted by a real server (e.g. the array casts in `_array_cast`) is not covered here;
see the live tests in `tests/test_postgres.py`.
"""

from __future__ import annotations

import datetime
from typing import cast

import pytest

from postbound import ColumnReference, TableReference
from postbound.db import Histogram
from postbound.postgres import PostgresDatabase
from postbound.postgres._pg import PostgresStatistics, _bucket_position
from tests.doubles import FakeDatabase, ScriptedCursor

# -- fixtures -----------------------------------------------------------------------------------------------

VOTES_BOUNTYAMOUNT = ColumnReference("bountyamount", TableReference("votes"))
POSTS_ANSWERCOUNT = ColumnReference("answercount", TableReference("posts"))
POSTS_CLOSEDDATE = ColumnReference("closeddate", TableReference("posts"))

# Stats database, captured with the exact queries PostgresStatistics issues, e.g.
#   SELECT reltuples FROM pg_class WHERE oid = 'votes'::regclass
#   SELECT null_frac FROM pg_stats WHERE tablename = 'votes' AND attname = 'bountyamount'
#   SELECT UNNEST(most_common_vals::text::int[]), UNNEST(most_common_freqs) FROM pg_stats WHERE ...
#   SELECT UNNEST(histogram_bounds::text::int[]) FROM pg_stats WHERE ...
# Both MCVs fall into the same histogram bucket (25, 125], and the first bucket (25, 25] is a duplicate bound.
# True value distribution (non-NULL rows only): 0: 2, 25: 74, 50: 1225, 75: 1, 100: 350, 125: 1, 150: 37, 200: 35,
# 250: 5, 300: 6, 350: 1, 400: 3, 500: 4
BOUNTYAMOUNT_CATALOG = {
    "reltuples": [(328064.0,)],
    "select null_frac": [(0.9948,)],
    "most_common_vals": [(50, 0.0036333334), (100, 0.0012)],
    "histogram_bounds": [(25,), (25,), (125,), (200,), (300,)],
    "information_schema.columns": [("integer",)],
}

# Stats database, captured like BOUNTYAMOUNT_CATALOG. The MCVs are listed by frequency, which is *not* their value
# order (1 before 0), and they all lie below the smallest histogram bound 9. The bounds contain several duplicates.
ANSWERCOUNT_CATALOG = {
    "reltuples": [(91976.0,)],
    "select null_frac": [(0.5327,)],
    "most_common_vals": [
        (1, 0.1956),
        (0, 0.151),
        (2, 0.07886667),
        (3, 0.026133334),
        (4, 0.009),
        (5, 0.0026666666),
        (6, 0.0016333334),
        (7, 0.0007),
        (8, 0.0007),
    ],
    "histogram_bounds": [(9,), (9,), (10,), (10,), (10,), (11,), (12,), (14,), (14,), (19,), (42,)],
    "information_schema.columns": [("integer",)],
}

# Stats database: SELECT n_distinct FROM pg_stats WHERE tablename = 'posts' AND attname = 'closeddate'
# A negative n_distinct is a fraction of the total row count.
CLOSEDDATE_CATALOG = {
    "reltuples": [(91976.0,)],
    "select n_distinct": [(-0.018066645,)],
}


def stats_over(catalog: dict) -> PostgresStatistics:
    cursor = ScriptedCursor(catalog)
    return PostgresStatistics(cast(PostgresDatabase, FakeDatabase(cursor_double=cursor)))


# -- num_distinct -------------------------------------------------------------------------------------------


def test_num_distinct_reads_a_positive_n_distinct_as_an_absolute_count() -> None:
    stats = stats_over({"select n_distinct": [(7.0,)]})

    assert stats.num_distinct(VOTES_BOUNTYAMOUNT) == 7


# -- most_common_values -------------------------------------------------------------------------------------


def test_most_common_values_are_sorted_by_descending_frequency() -> None:
    mcvs = stats_over(ANSWERCOUNT_CATALOG).most_common_values(POSTS_ANSWERCOUNT)

    assert mcvs is not None
    assert [val for val, _ in mcvs] == [1, 0, 2, 3, 4, 5, 6, 7, 8]


def test_most_common_values_are_absent_without_a_catalog_entry() -> None:
    stats = stats_over({"most_common_vals": [], "information_schema.columns": [("integer",)]})

    assert stats.most_common_values(VOTES_BOUNTYAMOUNT) is None


# -- histogram ----------------------------------------------------------------------------------------------


def test_histogram_splits_a_bucket_at_the_mcvs_it_contains() -> None:
    """The bucket (25, 125] holds ~30 non-MCV rows. MCV 50 lies at 1/4 and MCV 100 at 3/4 of the bucket, so the
    (uniformly spread) non-MCV rows are distributed 1/4 : 2/4 : 1/4 among (25, 50], (50, 100] and (100, 125], and each
    MCV adds its own frequency to the part it closes.
    """
    hist = stats_over(BOUNTYAMOUNT_CATALOG).histogram(VOTES_BOUNTYAMOUNT)

    assert hist == Histogram(
        [25, 50, 100, 125, 200, 300], [30, 1199, 409, 8, 30, 30], lower=25, bucket_interpolation="approx-uni"
    )


def test_histogram_puts_mcvs_below_the_histogram_range_in_front_of_it() -> None:
    hist = stats_over(ANSWERCOUNT_CATALOG).histogram(POSTS_ANSWERCOUNT)

    assert hist is not None
    assert hist.bounds == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 19, 42]
    assert hist.lower == 0


def test_histogram_collapses_duplicate_bounds_into_one_bucket() -> None:
    """Bound 10 occurs three times in the catalog, i.e. it closes three of the 10 equi-depth buckets of ~9.24 rows."""
    hist = stats_over(ANSWERCOUNT_CATALOG).histogram(POSTS_ANSWERCOUNT)

    assert hist is not None
    assert dict(hist)[10] == 28


def test_histogram_places_mcvs_above_the_histogram_range_behind_it() -> None:
    """Synthetic input: the real cases of MCVs above the histogram (e.g. movie_info.info_type_id in JOB) only occur in
    columns with 100 bounds, which are too large to be readable here.
    """
    catalog = {
        "reltuples": [(100.0,)],
        "select null_frac": [(0.0,)],
        "most_common_vals": [(99, 0.5)],
        "histogram_bounds": [(0,), (10,), (20,)],
        "information_schema.columns": [("integer",)],
    }

    hist = stats_over(catalog).histogram(VOTES_BOUNTYAMOUNT)

    assert hist == Histogram([10, 20, 99], [25, 25, 50], lower=0, bucket_interpolation="approx-uni")


def test_histogram_is_absent_without_histogram_bounds() -> None:
    stats = stats_over({"histogram_bounds": [], "information_schema.columns": [("integer",)]})

    assert stats.histogram(VOTES_BOUNTYAMOUNT) is None


# -- _bucket_position ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "lower", "upper", "expected"),
    [
        (50, 25, 125, 0.25),
        (datetime.date(2020, 1, 11), datetime.date(2020, 1, 1), datetime.date(2020, 1, 21), 0.5),
        ("m", "a", "z", 0.5),
        (10, 25, 125, 0.0),
        (200, 25, 125, 1.0),
    ],
    ids=["numeric", "date", "text-has-no-arithmetic", "below-bucket", "above-bucket"],
)
def test_bucket_position_interpolates_uniformly(value: object, lower: object, upper: object, expected: float) -> None:
    assert _bucket_position(value, lower, upper) == pytest.approx(expected)


# -- regression tests --------------------------------------------------------------------------------------


def test_num_distinct_scales_a_negative_n_distinct_to_an_integer() -> None:
    """Regression guard for the NULL rework of the statistics catalog: for a negative *n_distinct*, the fraction was
    multiplied with the `Cardinality` from `total_rows`. First this returned a float (1661.69...), later
    ``-1 * Cardinality`` raised because cardinalities cannot be negative.
    """
    n_distinct = stats_over(CLOSEDDATE_CATALOG).num_distinct(POSTS_CLOSEDDATE)

    assert n_distinct == 1662
    assert isinstance(n_distinct, int)


def test_most_common_values_scale_frequencies_with_all_rows_including_nulls() -> None:
    """Regression guard for the NULL rework of the statistics catalog: *most_common_freqs* are relative to all rows
    (including NULLs), but were scaled with the non-NULL rows only. For bountyamount (99.5% NULLs), value 50 occurs
    1225 times, but was reported with a frequency of 6.
    """
    mcvs = stats_over(BOUNTYAMOUNT_CATALOG).most_common_values(VOTES_BOUNTYAMOUNT)

    assert mcvs is not None
    assert list(mcvs) == [(50, 1192), (100, 394)]
    assert all(type(freq) is int for _, freq in mcvs)


def test_histogram_merges_the_mcvs_in_value_order() -> None:
    """Regression guard for the histogram reconstruction: the merge of histogram bounds and MCVs iterated the MCVs in
    frequency order (as `MostCommonValues` sorts them), but treated them as if they were sorted by value. For
    answercount, the MCVs [1, 0, 2, ...] produced the non-monotone bounds [1, 0, 2, ...].
    """
    hist = stats_over(ANSWERCOUNT_CATALOG).histogram(POSTS_ANSWERCOUNT)

    assert hist is not None
    assert hist.bounds == sorted(set(hist.bounds))


def test_histogram_does_not_count_the_mcv_rows_twice() -> None:
    """Regression guard for the histogram reconstruction: the bucket frequency was derived from all non-NULL rows, even
    though the histogram only describes the rows that are not covered by the MCVs. Since the MCVs were merged in with
    their own frequency, they were counted twice (comments.userid: 254,600 histogram rows vs. 171,470 non-NULL rows).
    """
    hist = stats_over(ANSWERCOUNT_CATALOG).histogram(POSTS_ANSWERCOUNT)

    assert hist is not None
    non_null_rows = (1 - 0.5327) * 91976
    assert hist.n_rows == pytest.approx(non_null_rows, abs=10)


def test_histogram_derives_n_buckets_from_n_plus_one_bounds() -> None:
    """Regression guard for the histogram reconstruction: the *N + 1* catalog bounds describe *N* equi-depth buckets,
    but the rows were divided by *N + 1*, and the smallest bound received a bucket of its own, even though it is only
    the lower bound of the first bucket. Without MCVs, the 1706 non-NULL rows of bountyamount spread over the
    4 buckets [25, 25], (25, 125], (125, 200] and (200, 300] as [682, 341, 341, 341] instead of ~426 each.
    """
    catalog = BOUNTYAMOUNT_CATALOG | {"most_common_vals": []}

    hist = stats_over(catalog).histogram(VOTES_BOUNTYAMOUNT)

    assert hist == Histogram([25, 125, 200, 300], [426, 427, 426, 427], lower=25, bucket_interpolation="approx-uni")


def test_histogram_keeps_a_lower_bound_of_zero() -> None:
    """Regression guard for the histogram reconstruction: the lower bound was computed as
    ``min(filter(None, [cur_hist, cur_mcv_val]))``, which drops falsy values. For answercount, the smallest value 0 was
    ignored and the lower bound became 1.
    """
    hist = stats_over(ANSWERCOUNT_CATALOG).histogram(POSTS_ANSWERCOUNT)

    assert hist is not None
    assert hist.lower == 0


def test_array_cast_supports_timestamp_columns() -> None:
    """Regression guard: `_array_cast` had no converter for timestamps, so the MCVs and the histogram of e.g.
    posts.closeddate in Stats raised a `ValueError`.
    """
    stats = stats_over({"information_schema.columns": [("timestamp without time zone",)]})

    assert stats._array_cast(POSTS_CLOSEDDATE) == "timestamp[]"


def test_histogram_rounding_preserves_the_total_number_of_rows() -> None:
    """Regression guard for the histogram reconstruction: rounding every bucket frequency individually accumulates
    the rounding errors. posts.closeddate in Stats has ~16.6 rows in each of its 100 buckets, which added up to 1700
    instead of 1662 rows.
    """
    catalog = {
        "reltuples": [(100.0,)],
        "select null_frac": [(0.0,)],
        "most_common_vals": [],
        "histogram_bounds": [(0,), (1,), (2,), (3,), (4,), (5,), (6,)],
        "information_schema.columns": [("integer",)],
    }

    hist = stats_over(catalog).histogram(VOTES_BOUNTYAMOUNT)

    assert hist is not None
    assert hist.n_rows == 100
