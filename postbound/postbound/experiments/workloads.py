from __future__ import annotations

import collections
import pathlib
import random

import natsort

from postbound.qal import qal, parser

workloads_base_dir = "../../../workloads"


class Workload(collections.UserDict[str, qal.SqlQuery]):
    @staticmethod
    def read(root_dir: str, *, query_file_pattern: str = "*.sql", name: str = "", label_prefix: str = "") -> "Workload":
        queries: dict[str, qal.SqlQuery] = {}
        root = pathlib.Path(root_dir)

        for query_file_path in root.glob(query_file_pattern):
            with open(query_file_path, "r", encoding="utf-8") as query_file:
                raw_contents = query_file.readlines()
            query_contents = "\n".join([line for line in raw_contents])
            parsed_query = parser.parse_query(query_contents)
            query_label = query_file_path.stem
            queries[label_prefix + query_label] = parsed_query

        return Workload(queries, name=name, root=root)

    def __init__(self, queries: dict[str, qal.SqlQuery], name: str = "", root: pathlib.Path = None):
        super().__init__(queries)
        self._name = name
        self._root = root
        self._sorted_labels = natsort.natsorted(list(self.keys()))
        self._sorted_queries = [self.data[label] for label in self._sorted_labels]

    def queries(self) -> list[qal.SqlQuery]:
        return list(self._sorted_queries)

    def first(self, n: int) -> "Workload":
        first_n_labels = self._sorted_labels[:n]
        sub_workload = {label: self.data[label] for label in first_n_labels}
        return Workload(sub_workload, self._name, self._root)

    def pick_random(self, n: int) -> "Workload":
        labels = list(self.keys())
        selected_labels = random.sample(labels, n)
        sub_workload = {label: self.data[label] for label in selected_labels}
        return Workload(sub_workload, self._name, self._root)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        name = self._name if self._name else self._root.stem
        return f"Workload: {name} ({len(self)} queries)"


def job() -> Workload:
    return Workload.read(f"{workloads_base_dir}/JOB-Queries/implicit", name="JOB")


def ssb() -> Workload:
    return Workload.read(f"{workloads_base_dir}/SSB-Queries", name="SSB")


def stack() -> Workload:
    stack_root = pathlib.Path(f"{workloads_base_dir}/Stack-Queries")
    merged_queries = {}
    for query_container in stack_root.iterdir():
        if not query_container.is_dir():
            continue
        sub_workload = Workload.read(str(query_container), label_prefix=query_container.stem + "/")
        merged_queries |= sub_workload.data
    merged_workload = Workload(merged_queries, "Stack", stack_root)
    return merged_workload
