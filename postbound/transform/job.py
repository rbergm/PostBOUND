
import pathlib

from transform import mosp


def load_query(label: str, *, src_dir: str = "../workloads/JOB-Queries/implicit") -> mosp.MospQuery:
    fname = f"{label}.sql"
    path = pathlib.Path(src_dir) / fname
    with open(path, "r", encoding="utf-8") as query_file:
        raw_query = "".join(query_file.readlines())
    return mosp.MospQuery.parse(raw_query)
