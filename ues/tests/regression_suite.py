
import numbers
import pathlib

from typing import Dict


def load_job_workload(path: str = "../../workloads/JOB-Queries/implicit",
                      source_pattern: str = "*.sql") -> Dict[str, str]:
    root = pathlib.Path(path)
    workload_files = list(root.glob(source_pattern))
    queries = {}
    for query_file in workload_files:
        with open(query_file, "r") as raw_query:
            lines = raw_query.readlines()
            single_line = " ".join(line.strip() for line in lines)
            queries[query_file.stem] = single_line
    return queries


def assert_less_equal(smaller: numbers.Number, larger: numbers.Number, msg: str = "", tolerance: float = 0.99):
    """Asserts that the smaller <= larger, but allowing a certain tolerance.

    `tolerance` should be a value <= 1 and will shift the smaller number accordingly. For example, if the smaller
    number is allowed to be up to 1% larger than the larger number (which also is the default setting), tolerance
    should be set to 0.99. If no tolerance is allowed, use 1.
    """
    try:
        assert smaller * tolerance <= larger
    except AssertionError as e:
        user_msg = f" : {msg}" if msg else ""
        raise AssertionError(f"AssertionError: {smaller} not less than or equal to {larger}" + user_msg)
