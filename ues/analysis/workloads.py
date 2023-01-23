
import collections
import pathlib
from typing import Dict, List

from transform import mosp, util


class Workload(collections.UserDict):
    @staticmethod
    def read(root_dir: str, *, query_file_pattern: str = "*.sql", name: str = "", label_prefix: str = "") -> "Workload":
        queries: Dict[str, str] = {}
        root = pathlib.Path(root_dir)

        for query_file_path in root.glob(query_file_pattern):
            with open(query_file_path, "r", encoding="utf-8") as query_file:
                raw_contents = query_file.readlines()
            query_contents = "\n".join([line for line in raw_contents])
            parsed_query = mosp.MospQuery.parse(query_contents)
            query_label = query_file_path.stem
            queries[label_prefix + query_label] = parsed_query

        return Workload(queries, name=name, root=root)

    def __init__(self, queries: Dict[str, str], name: str = "", root: pathlib.Path = None):
        super().__init__(queries)
        self._name = name
        self._root = root

    def queries(self) -> List[mosp.MospQuery]:
        return list(self.data.values())

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        name = self._name if self._name else self._root.stem
        return f"Workload: {name} ({len(self)} queries)"


def job() -> Workload:
    return Workload.read("../workloads/JOB-Queries/implicit", name="JOB")


def ssb() -> Workload:
    return Workload.read("../workloads/SSB-Queries", name="SSB")

def stack() -> Workload:
    stack_root = pathlib.Path("../workloads/Stack-Queries")
    merged_queries = {}
    for query_container in stack_root.iterdir():
        if not query_container.is_dir():
            continue
        print("... Reading", query_container.stem)
        sub_workload = Workload.read(str(query_container), label_prefix=query_container.stem + "/")
        print("... Merging")
        merged_queries = util.dict_merge(merged_queries, sub_workload.data)
    merged_workload = Workload(merged_queries, "Stack", stack_root)
    return merged_workload
