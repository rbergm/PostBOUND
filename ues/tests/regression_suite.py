
import collections
import numbers
import pathlib
from typing import Any, List, Tuple

from typing import Dict


def __load_workload(path: str, source_pattern: str, *,
                    label_prefix: str = "") -> Dict[str, str]:
    root = pathlib.Path(path)
    workload_files = list(root.glob(source_pattern))
    queries = {}
    for query_file in workload_files:
        with open(query_file, "r", encoding="utf-8") as raw_query:
            lines = raw_query.readlines()
            query = "\n".join(line.strip() for line in lines)
            queries[label_prefix + query_file.stem] = query
    return queries


def load_job_workload(path: str = "../../workloads/JOB-Queries/implicit",
                      source_pattern: str = "*.sql") -> Dict[str, str]:
    return __load_workload(path, source_pattern)


def load_ssb_workload(path: str = "../../workloads/SSB-Queries", source_pattern: str = "*.sql") -> Dict[str, str]:
    return __load_workload(path, source_pattern)


def load_stack_workload(path: str = "../../workloads/Stack-Queries", source_pattern: str = "*.sql") -> Dict[str, str]:
    stack_directory = pathlib.Path(path)
    workload = dict()
    for workload_dir in stack_directory.glob("*/**"):
        sub_workload = __load_workload(str(workload_dir), source_pattern,
                                       label_prefix=str(workload_dir.name) + "/")
        for label, query in sub_workload.items():
            workload[label] = query
    return workload


def assert_less_equal(smaller: numbers.Number, larger: numbers.Number, msg: str = "", tolerance: float = 0.99):
    """Asserts that the smaller <= larger, but allowing a certain tolerance.

    `tolerance` should be a value <= 1 and will shift the smaller number accordingly. For example, if the smaller
    number is allowed to be up to 1% larger than the larger number (which also is the default setting), tolerance
    should be set to 0.99. If no tolerance is allowed, use 1.
    """
    try:
        assert smaller * tolerance <= larger
    except AssertionError:
        user_msg = f" : {msg}" if msg else ""
        raise AssertionError(f"AssertionError: {smaller} not less than or equal to {larger}" + user_msg)


def assert_result_sets_equal(first_set: List[Tuple[Any]], second_set: List[Tuple[Any]], *, ordered: bool = False):
    if type(first_set) != type(second_set):
        raise AssertionError(f"Result sets have different types: {type(first_set)} and {type(second_set)}")
    elif not isinstance(first_set, list):
        if first_set != second_set:
            raise AssertionError(f"Result sets differ: {first_set} and {second_set}")
        return

    if len(first_set) != len(second_set):
        raise AssertionError("Result sets are not of equal length!")

    first_set = [tuple(tup) for tup in first_set]
    second_set = [tuple(tup) for tup in second_set]

    if ordered:
        for tuple_idx in range(0, len(first_set)):
            first_tuple = first_set[tuple_idx]
            second_tuple = second_set[tuple_idx]
            if first_tuple != second_tuple:
                raise AssertionError(f"Tuple {first_tuple} does not equal second tuple {second_tuple}!")
    else:
        first_set_counter = collections.defaultdict(int)
        second_set_counter = collections.defaultdict(int)

        for tup in first_set:
            first_set_counter[tup] += 1
        for tup in second_set:
            second_set_counter[tup] += 1

        first_set = set(first_set)
        second_set = set(second_set)

        for tup in first_set:
            if tup not in second_set:
                raise AssertionError(f"Tuple {tup} from first set has no partner in second set!")
        for tup in second_set:
            if tup not in first_set:
                raise AssertionError(f"Tuple {tup} from second set has no partner in first set!")

        for tup, first_tuple_counter in first_set_counter.items():
            second_tuple_counter = second_set_counter[tup]
            if first_tuple_counter != second_tuple_counter:
                raise AssertionError(f"Tuple {tup} appears {first_tuple_counter} times in first set, but "
                                     f"{second_tuple_counter} times in second set!")
