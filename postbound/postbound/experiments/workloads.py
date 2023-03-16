from __future__ import annotations

import collections
import copy
import pathlib
import random
import typing
from typing import Hashable, Iterable, Optional

import natsort

from postbound.qal import qal, parser
from postbound.util import dicts as dict_utils

workloads_base_dir = "../../../workloads"

LabelType = typing.TypeVar("LabelType", bound=Hashable)


class Workload(collections.UserDict[LabelType, qal.SqlQuery]):
    @staticmethod
    def read(root_dir: str, *, query_file_pattern: str = "*.sql", name: str = "",
             label_prefix: str = "", file_encoding: str = "utf-8") -> Workload[str]:
        queries: dict[str, qal.SqlQuery] = {}
        root = pathlib.Path(root_dir)

        for query_file_path in root.glob(query_file_pattern):
            with open(query_file_path, "r", encoding=file_encoding) as query_file:
                raw_contents = query_file.readlines()
            query_contents = "\n".join([line for line in raw_contents])
            parsed_query = parser.parse_query(query_contents)
            query_label = query_file_path.stem
            queries[label_prefix + query_label] = parsed_query

        return Workload(queries, name=name, root=root)

    def __init__(self, queries: dict[LabelType, qal.SqlQuery], name: str = "", root: Optional[pathlib.Path] = None):
        super().__init__(queries)
        self._name = name
        self._root = root

        self._sorted_labels = natsort.natsorted(list(self.keys()))
        self._sorted_queries = []
        self._update_query_order()

        self._label_mapping = dict_utils.invert(self.data)

    def queries(self) -> list[qal.SqlQuery]:
        return list(self._sorted_queries)

    def labels(self) -> list[LabelType]:
        return list(self._sorted_labels)

    def entries(self) -> list[tuple[LabelType, qal.SqlQuery]]:
        return list(zip(self._sorted_labels, self._sorted_queries))

    def label_of(self, query: qal.SqlQuery) -> LabelType:
        return self._label_mapping[query]

    def first(self, n: int) -> Workload[LabelType]:
        first_n_labels = self._sorted_labels[:n]
        sub_workload = {label: self.data[label] for label in first_n_labels}
        return Workload(sub_workload, self._name, self._root)

    def pick_random(self, n: int) -> Workload[LabelType]:
        labels = list(self.keys())
        selected_labels = random.sample(labels, n)
        sub_workload = {label: self.data[label] for label in selected_labels}
        return Workload(sub_workload, self._name, self._root)

    def shuffle(self) -> Workload[LabelType]:
        shuffled_workload = copy.copy(self)
        shuffled_workload._sorted_labels = random.sample(self._sorted_labels, k=len(self))
        shuffled_workload._update_query_order()
        return shuffled_workload

    def _update_query_order(self) -> None:
        self._sorted_queries = [self.data[label] for label in self._sorted_labels]

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self._name:
            return f"Workload: {self._name} ({len(self)} queries)"
        elif self._root:
            return f"Workload: {self._root.stem} ({len(self)} queries)"
        else:
            return f"Workload: {len(self)} queries"


def read_workload(path: str, name: str = "", *, query_file_pattern: str = "*.sql",
                  recurse_subdirectories: bool = False, query_label_prefix: str = "",
                  file_encoding: str = "utf-8") -> Workload[str]:
    base_dir_workload = Workload.read(path, name=name, query_file_pattern=query_file_pattern,
                                      label_prefix=query_label_prefix, file_encoding=file_encoding)
    if not recurse_subdirectories:
        return base_dir_workload

    merged_queries = dict(base_dir_workload.data)
    root_dir = pathlib.Path(path)
    for subdir in root_dir.iterdir():
        if not subdir.is_dir():
            continue
        subdir_prefix = ((query_label_prefix + "/") if query_label_prefix and not query_label_prefix.endswith("/")
                         else query_label_prefix)
        subdir_prefix += subdir.stem + "/"
        subdir_workload = read_workload(str(subdir), query_file_pattern=query_file_pattern, recurse_subdirectories=True,
                                        query_label_prefix=subdir_prefix)
        merged_queries |= subdir_workload.data
    return Workload(merged_queries, name, root_dir)


def generate_workload(queries: Iterable[qal.SqlQuery], *, name: str = "",
                      labels: Optional[dict[qal.SqlQuery, LabelType]] = None,
                      workload_root: Optional[pathlib.Path] = None) -> Workload[LabelType]:
    if not labels:
        labels: dict[qal.SqlQuery, int] = {query: idx + 1 for idx, query in enumerate(queries)}
    workload_contents = dict_utils.invert(labels)
    return Workload(workload_contents, name, workload_root)


def job(file_encoding: str = "utf-8") -> Workload[str]:
    return Workload.read(f"{workloads_base_dir}/JOB-Queries/implicit", name="JOB", file_encoding=file_encoding)


def ssb(file_encoding: str = "utf-8") -> Workload[str]:
    return Workload.read(f"{workloads_base_dir}/SSB-Queries", name="SSB", file_encoding=file_encoding)


def stack(file_encoding: str = "utf-8") -> Workload[str]:
    return read_workload(f"{workloads_base_dir}/Stack-Queries", "Stack", recurse_subdirectories=True,
                         file_encoding=file_encoding)
