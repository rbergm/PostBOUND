import abc
import unittest


def _rebuild_result_set(result_set: object) -> list[tuple[object]]:
    """Transforms a possibly simplified result set as returned by the Database interface back to a normalized one."""

    # case 1: result set of a single row -- ('foo', 42) becomes [('foo', 42)]
    if isinstance(result_set, tuple):
        return [result_set]

    # case 2: result set of a single row of a single value -- 42 becomes [(42,)]
    if not isinstance(result_set, list):
        return [(result_set,)]

    first_row = result_set[0]
    # case 3: result set of multiple rows of multiple values is left as-is
    if isinstance(first_row, tuple):
        return result_set

    # case 4: result set of multiple rows of single values -- ['foo', 'bar'] becomes [('foo',), ('bar',)]
    return [(value,) for value in result_set]


def _stringify_result_set(result_set: list[tuple[object]]) -> str:
    if not "__len__" in dir(result_set):
        return str(result_set)
    if len(result_set) > 5:
        shortened_result_set = result_set[:5]
        return f"result set ({len(result_set)} elements total) :: first 5 = {shortened_result_set}"
    return f"result set ({len(result_set)} elements) :: contents = {result_set}"


def _assert_default_result_sets_equal(first_set: list[tuple[object]], second_set: list[tuple[object]]) -> None:
    first_set_set = set(first_set)
    second_set_set = set(second_set)
    if first_set_set != second_set_set:
        first_set_str = _stringify_result_set(first_set)
        second_set_str = _stringify_result_set(second_set)
        raise AssertionError(f"Result sets differ: {first_set_str} vs. {second_set_str}")


def _assert_ordered_result_sets_equal(first_set: list[tuple[object]], second_set: list[tuple[object]]) -> None:
    for cursor, row in enumerate(first_set):
        comparison_row = second_set[cursor]
        if row != comparison_row:
            first_set_str = _stringify_result_set(first_set)
            second_set_str = _stringify_result_set(second_set)
            raise AssertionError(f"Result sets differ: {first_set_str} vs {second_set_str}")


class DatabaseTestCase(unittest.TestCase, abc.ABC):
    def assertResultSetsEqual(self, first_set: object, second_set: object, *, ordered: bool = False) -> None:
        if type(first_set) != type(second_set):
            error_msg = "Result sets have different types: "
            first_set_str = _stringify_result_set(first_set)
            second_set_str = _stringify_result_set(second_set)
            error_msg += f"{first_set_str} ({type(first_set)}) and {second_set_str} ({type(second_set)})"
            raise AssertionError(error_msg)

        first_set = _rebuild_result_set(first_set)
        second_set = _rebuild_result_set(second_set)
        if len(first_set) != len(second_set):
            first_set_str = _stringify_result_set(first_set)
            second_set_str = _stringify_result_set(second_set)
            raise AssertionError(f"Result sets have different length: {first_set_str} and {second_set_str}")

        if ordered:
            _assert_ordered_result_sets_equal(first_set, second_set)
        else:
            _assert_default_result_sets_equal(first_set, second_set)


class QueryTestCase(unittest.TestCase, abc.ABC):

    def assertQueriesEqual(self, first_query: object, second_query: object, message: str = ""):
        first_query = str(first_query).strip().removesuffix(";").lower()
        second_query = str(second_query).strip().removesuffix(";").lower()
        return self.assertEqual(first_query, second_query, message)
