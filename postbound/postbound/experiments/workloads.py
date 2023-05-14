"""Utilities to read sets of queries and operate on them conveniently.

In addition, this module provides methods to read some pre-defined workloads directly. These include the
Join Order Benchmark [0], Star Schema Benchmark [1] and Stack Benchmark [2]. For these utilities to work, PostBOUND
assumes that there is a directory which contains the actual queries available somewhere. The precise location can be
customized via the `workloads_base_dir` variable.

The expected directory layout is the following:

```
/ <workloads_base_dir>
  +- JOB-Queries/
    +- <queries>
  +- SSB-Queries/
    +- <queries>
  +- Stack-Queries/
    +- <queries>
```

The GitHub repository of PostBOUND should contain such a directory by default.

---

[0] Viktor Leis et al.: How Good Are Query Optimizers, Really? (Proc. VLDB Endow. 9, 3 (2015))

[1] Patrick E. O’Neil et al.: The Star Schema Benchmark and Augmented Fact Table Indexing. (TPCTC’2009)

[2] Ryan Marcus et al.: Bao: Making Learned Query Optimization Practical. (SIGMOD'2021)
"""
from __future__ import annotations

import collections
import copy
import pathlib
import random
import typing
from collections.abc import Callable, Iterable
from typing import Hashable, Optional

import natsort
import pandas as pd

from postbound.qal import qal, parser
from postbound.util import dicts as dict_utils

workloads_base_dir = "../workloads"
"""Indicates the PostBOUND directory that contains all natively supported workloads.

Can be changed to match the project-specific file layout.
"""

LabelType = typing.TypeVar("LabelType", bound=Hashable)


class Workload(collections.UserDict[LabelType, qal.SqlQuery]):
    """A `Workload` collects a number of queries and provides utilities to operate on them conveniently.

    In addition to the actual queries, each query is annotated by a label that can be used to retrieve the query more
    efficiently. Labels can be arbitrary types as long as they are hashable. Since the workload inherits from dict,
    the label can be used directly to fetch the associated query.

    Each workload can be given a name, which is mainly intended for readability in __str__ methods and does not serve
    a functional purpose.

    During iteration, queries will typically be returned in order according to the natural order of the query labels.
    """

    @staticmethod
    def read(root_dir: str, *, query_file_pattern: str = "*.sql", name: str = "",
             label_prefix: str = "", file_encoding: str = "utf-8") -> Workload[str]:
        """Reads all SQL queries from `root_dir` whose file name matches the `query_file_pattern`.

        This method assumes that each query file contains exactly one query.

        Query labels will be constructed based on the file name of the source files. For example, a query contained in
        file `q-1-1.sql` will receive label `q-1-1` (note that the trailing file extension is cropped). If a
        `label_prefix` is given, it will be inserted before the file name-based label.

        All query files will be read with the given `file_encoding`.

        The resulting workload can be annotated by a name.
        """
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
        """Provides all queries in the workload in natural order (according to their labels)."""
        return list(self._sorted_queries)

    def labels(self) -> list[LabelType]:
        """Provides all query labels of the workload in natural order."""
        return list(self._sorted_labels)

    def entries(self) -> list[tuple[LabelType, qal.SqlQuery]]:
        """Provides all (label, query) pairs in the workload, in natural order of the query labels."""
        return list(zip(self._sorted_labels, self._sorted_queries))

    def head(self) -> Optional[tuple[LabelType, qal.SqlQuery]]:
        """Provides the first query in the workload."""
        if not self._sorted_labels:
            return None
        return self._sorted_labels[0], self._sorted_queries[0]

    def label_of(self, query: qal.SqlQuery) -> LabelType:
        """Provides the label of the given query."""
        return self._label_mapping[query]

    def first(self, n: int) -> Workload[LabelType]:
        """Provides the first `n` queries of the workload, according to the natural order of the query labels.

        If there are less than `n` queries in the workload, all queries will be returned.
        """
        first_n_labels = self._sorted_labels[:n]
        sub_workload = {label: self.data[label] for label in first_n_labels}
        return Workload(sub_workload, self._name, self._root)

    def pick_random(self, n: int) -> Workload[LabelType]:
        """Provides `n` queries from the workload, which are chosen at random.

        If there are less than `n` queries in the workload, all queries will be returned.
        """
        n = min(n, len(self._sorted_queries))
        selected_labels = random.sample(self._sorted_labels, n)
        sub_workload = {label: self.data[label] for label in selected_labels}
        return Workload(sub_workload, self._name, self._root)

    def with_prefix(self, label_prefix: LabelType) -> Workload[LabelType]:
        """Returns all queries from the workload, whose label starts with the given `label_prefix`.

        This method requires that all label instances provide a `startswith` method (as is the case for simple string
        labels).
        """
        if "startswith" not in dir(label_prefix):
            raise ValueError("label_prefix must have startswith() method")
        prefix_queries = {label: query for label, query in self.data.items() if label.startswith(label_prefix)}
        return Workload(prefix_queries, name=self._name, root=self._root)

    def filter_by(self, predicate: Callable[[LabelType, qal.SqlQuery], bool]) -> Workload[LabelType]:
        """Provides all queries from the workload that match the given predicate."""
        matching_queries = {label: query for label, query in self.data.items() if predicate(label, query)}
        return Workload(matching_queries, name=self._name, root=self._root)

    def shuffle(self) -> Workload[LabelType]:
        """Randomly changes the order of the queries in the workload."""
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
    """Reads workloads queries from the base directory indicated by `path` to create a new `Workload` object.

    In contrast to the `Workload.read` method, this function also supports recursive directory layouts: if the
    `recurse_subdirectories` parameter is set to `True`, this method will read all files matching the
    `query_file_pattern` from `path` first. Afterwards, it will do the same for all subdirectories (and continue
    to do so with their subdirectories recursively). In each recursion step the name of the current subdirectory
    will be used as a label prefix, i.e. a query `q-1.sql` in subdirectory `base` will receive the label `base/q-1`.
    """
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
        subdir_workload = read_workload(str(subdir), query_file_pattern=query_file_pattern,
                                        recurse_subdirectories=True, query_label_prefix=subdir_prefix)
        merged_queries |= subdir_workload.data
    return Workload(merged_queries, name, root_dir)


def read_batch_workload(filename: str, name: str = "", *, file_encoding: str = "utf-8") -> Workload[int]:
    """Reads all the queries from a single file to construct a new `Workload` object.

    The input file has to contain one valid SQL query per line. While empty lines are skipped, any non-SQL line will
    raise an Error.

    The workload will have numeric labels: the query in the first line will have label 1, the second one label 2 and
    so on.

    The `name` parameter can be used to give the resulting workload a better name. If omitted, this will default to
    the supplied filename.
    """
    filename = pathlib.Path(filename)
    name = name if name else filename.stem
    with open(filename, "r", encoding=file_encoding) as query_file:
        raw_queries = query_file.readlines()
        parsed_queries = [parser.parse_query(raw) for raw in raw_queries if raw]
        return generate_workload(parsed_queries, name=name, workload_root=filename)


def read_csv_workload(filename: str, name: str = "", *, query_column: str = "query",
                      label_column: Optional[str] = None, file_encoding: str = "utf-8") -> Workload:
    """Reads all the queries from a CSV file to construct a new `Workload` object.

    The column containing the actual queries can be configured via the `query_column` parameter. Likewise, the
    CSV file can already provide query labels in the `label_column` column. If this parameter is omitted, labels will
    be inferred based on the row number.

    The `name` parameter can be used to give the resulting workload a better name. If omitted, this will default to
    the supplied filename.
    """
    filename = pathlib.Path(filename)
    name = name if name else filename.stem
    columns = [query_column] + [label_column] if label_column else []
    workload_df = pd.read_csv(filename, usecols=columns, converters={query_column: parser.parse_query},
                              encoding=file_encoding)

    queries = workload_df[query_column].tolist()
    if label_column:
        labels = workload_df[label_column].tolist()
        label_provider = dict(zip(queries, labels))
    else:
        label_provider = None

    return generate_workload(queries, name=name, labels=label_provider, workload_root=filename)


def generate_workload(queries: Iterable[qal.SqlQuery], *, name: str = "",
                      labels: Optional[dict[qal.SqlQuery, LabelType]] = None,
                      workload_root: Optional[pathlib.Path] = None) -> Workload[LabelType]:
    """Creates a `Workload` object for all given `queries`.

    Labels can optionally be provided with the `label` parameter. If this is omitted, labels will be inferred based
    on the query index, i.e. the first query in the iterable receives label `1`, the second one label `2` and so on.

    The `name` parameter can be used to give the resulting workload a better name. If omitted, this will be generated
    from the `workload_root` if possible.
    """
    name = name if name else (workload_root.stem if workload_root else "")
    if not labels:
        labels: dict[qal.SqlQuery, int] = {query: idx + 1 for idx, query in enumerate(queries)}
    workload_contents = dict_utils.invert(labels)
    return Workload(workload_contents, name, workload_root)


def job(file_encoding: str = "utf-8", *, simplified: bool = True) -> Workload[str]:
    """Provides an instance of the Join Order Benchmark.

    Queries will be read from the JOB directory relative to `workloads_base_dir`. The expected layout is:
    `<workloads_base_dir>/JOB-Queries/implicit/<queries>`.

    ---
    see: Viktor Leis et al.: How Good Are Query Optimizers, Really? (Proc. VLDB Endow. 9, 3 (2015))
    """
    simplified_dir = "/simplified" if simplified else ""
    return Workload.read(f"{workloads_base_dir}/JOB-Queries{simplified_dir}", name="JOB", file_encoding=file_encoding)


def ssb(file_encoding: str = "utf-8") -> Workload[str]:
    """Provides an instance of the Star Schema Benchmark.

    Queries will be read from the SSB directory relative to `workloads_base_dir`. The expected layout is:
    `<workloads_base_dir>/SSB-Queries/<queries>`.

    ---
    see: Patrick E. O’Neil et al.: The Star Schema Benchmark and Augmented Fact Table Indexing. (TPCTC’2009)
    """
    return Workload.read(f"{workloads_base_dir}/SSB-Queries", name="SSB", file_encoding=file_encoding)


def stack(file_encoding: str = "utf-8") -> Workload[str]:
    """Provides an instance of the Stack Benchmark.

    Queries will be read from the Stack directory relative to `workloads_base_dir`. The expected layout is:
    `<workloads_base_dir>/Stack-Queries/<sub-directories>/<queries>`. Alternatively, all queries can be contained in
    the `Stack-Queries` directory directly.

    ---
    see: Ryan Marcus et al.: Bao: Making Learned Query Optimization Practical. (SIGMOD'2021)
    """
    return read_workload(f"{workloads_base_dir}/Stack-Queries", "Stack", recurse_subdirectories=True,
                         file_encoding=file_encoding)
