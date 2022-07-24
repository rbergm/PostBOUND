
import pathlib

from typing import Dict


def load_job_workload(path: str = "../../simplicity-done-right/JOB-Queries/implicit",
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
