"""Provides abstractions to represent entire query workloads and utilities to read some pre-defined instances.

The main abstraction provided by this class is the `Workload`. A number of utility functions to read collections of queries
from different sources and input formats into workload objects exist as well. The pre-defined workloads include the
Join Order Benchmark [0]_, Star Schema Benchmark [1]_, Stack Benchmark [2]_ and Stats Benchmark [3]_. In order for the utility
functions that read those workloads to work, PostBOUND assumes that there is a directory which contains the actual queries
available somewhere. The precise location can be customized via the `workloads_base_dir` variable.

The expected directory layout is the following:

::

    <workloads_base_dir>
    ├╴JOB-Queries/
    │   └╴ <queries>
    ├╴ SSB-Queries/
    │   └╴ <queries>
    ├╴ Stack-Queries/
    │   └╴ <queries>
    ├╴ Stats-CEB/
    |   └╴ queries/
    |       └╴ <queries>


By default, PostBOUND assumes that the workload directory is contained one directory level higher than the root directory that
contains the PostBOUND source code. The GitHub repository of PostBOUND should have such a file layout by default.

References
----------

.. [0] Viktor Leis et al.: How Good Are Query Optimizers, Really? (Proc. VLDB Endow. 9, 3 (2015))
.. [1] Patrick E. O'Neil et al.: The Star Schema Benchmark and Augmented Fact Table Indexing. (TPCTC'2009)
.. [2] Ryan Marcus et al.: Bao: Making Learned Query Optimization Practical. (SIGMOD'2021)
.. [3] Yuxing Han et al.: Cardinality Estimation in DBMS: A Comprehensive Benchmark Evaluation (Proc. VLDB Endow. 15, 4 (2022))
"""

from __future__ import annotations

import collections
import os
import pathlib
import random
import typing
from collections.abc import Callable, Hashable, Iterable, Sequence
from typing import Optional

import natsort
import pandas as pd

from .. import util
from ..db._db import DatabasePool
from ..qal import parser
from ..qal._qal import SqlQuery

workloads_base_dir: str | pathlib.Path = ""
"""Indicates the PostBOUND directory that contains all natively supported workloads.

This setting can be changed to match the project-specific file layout.

By default, PostBOUND tries to guess the location of the workloads directory by checking the following locations in that order:

1. If there is a `workloads` directory in the current working directory, this is used.
2. If there is a `workloads` directory in the parent directory of the current working directory, this is used.
3. If there is a `workloads` directory in the parent directory of the experiments package, this is used.
   **This is the setting being triggered when installing PostBOUND as a Python package.**
4. If there is a `workloads` directory in the user's home directory, under `.local/share/postbound`, this is used.
"""

if pathlib.Path("workloads").is_dir():
    workloads_base_dir = pathlib.Path("workloads").absolute()
elif (pathlib.Path().parent / "workloads").is_dir():
    workloads_base_dir = pathlib.Path("../workloads").resolve()
elif (
    pathlib.Path(__file__).parent.parent / "workloads"
).is_dir():  # file into directory into parent
    # venv installation
    workloads_base_dir = (pathlib.Path(__file__).parent.parent / "workloads").resolve()
elif (pathlib.Path().home() / ".local" / "share" / "postbound" / "workloads").is_dir():
    # last resort: global workloads directory
    workloads_base_dir = (
        pathlib.Path().home() / ".local" / "share" / "postbound" / "workloads"
    ).resolve()


LabelType = typing.TypeVar("LabelType", bound=Hashable)
"""The labels that are used to identify individual queries in a workload."""

NewLabelType = typing.TypeVar("NewLabelType", bound=Hashable)
"""In case of mutations of the workload labels, this denotes the new type of the labels after the mutation."""


class Workload(collections.UserDict[LabelType, SqlQuery]):
    """A workload collects a number of queries (read: benchmark) and provides utilities to operate on them conveniently.

    In addition to the actual queries, each query is annotated by a label that can be used to retrieve the query more
    nicely. E.g. for queries in the Join Order Benchmark, access by their index is supported - such as ``job["1a"]``. Labels
    can be arbitrary types as long as they are hashable. Since the workload inherits from dict, the label can be used directly
    to fetch the associated query (and will raise ``KeyError`` instances for unknown labels).

    Each workload can be given a name, which is mainly intended for readability in ``__str__`` methods and does not serve
    a functional purpose. However, it may be good practice to use a normalized name that can be used in different contexts
    such as in file names, etc.

    When using methods that allow iteration over the queries, they will typically be returned in order according to the natural
    order of the query labels. However, since workloads can be shuffled randomly, this order can also be destroyed.

    A workload is implemented as an immutable data object. Therefore, it is not possible/not intended to change the contents
    of a workload object later on. All methods that mutate the contents instead provide new workload instances.

    Parameters
    ----------
    queries : dict[LabelType, SqlQuery]
        The queries that form the actual workload
    name : str, optional
        A name that can be used to identify or represent the workload, by default ``""``.
    root : Optional[pathlib.Path], optional
        The root directory that contains the workload queries. This is mainly used to somehow identify the workload when no
        name is given or the workload contents do not match the expected queries. Defaults to ``None``.

    Notes
    -----
    Workloads support many of the Python builtin-methods thanks to inheriting from ``UserDict``. Namely, the *len*, *iter* and
    *in* methods work as expected on the labels. Furthermore, multiple workload objects can be added, subtracted and
    intersected using set semantics. Subtraction and intersection also work based on individual labels.
    """

    @staticmethod
    def read(
        root_dir: str,
        *,
        query_file_pattern: str = "*.sql",
        name: str = "",
        label_prefix: str = "",
        file_encoding: str = "utf-8",
        bind_columns: bool = True,
    ) -> Workload[str]:
        """Reads all SQL queries from a specific directory into a workload object.

        This method assumes that the queries are stored in individual files, one query per file. The query labels will be
        constructed based on the file name of the source files. For example, a query contained in file ``q-1-1.sql`` will
        receive label ``q-1-1`` (note that the trailing file extension is dropped). If the `label_prefix` is given, it will be
        inserted before the file name-based label.

        Parameters
        ----------
        root_dir : str
            Directory containing the individual query files
        query_file_pattern : str, optional
            File name pattern that is shared by all query files. Only files matching the pattern will be read and each matching
            file is assumed to be a valid workload query. This is resolved as a glob expression. Defaults to ``"*.sql"``
        name : str, optional
            An optional name that can be used to identify the workload. Empty by default.
        label_prefix : str, optional
            A prefix to add before each query label. Empty by default. Notice that the prefix will be prepended as-is, i.e. no
            separator character is inserted. If a separator is desired, it has to be part of the prefix.
        file_encoding : str, optional
            The encoding of the query files. All files must share the same encoding. Defaults to UTF-8 encoding.

        Returns
        -------
        Workload[str]
            A workload consisting of all query files contained in the root directory.

        See Also
        --------
        pathlib.Path.glob
        """
        queries: dict[str, SqlQuery] = {}
        root = pathlib.Path(root_dir)

        for query_file_path in root.glob(query_file_pattern):
            with open(query_file_path, "r", encoding=file_encoding) as query_file:
                raw_contents = query_file.readlines()
            query_contents = "\n".join([line for line in raw_contents])
            try:
                parsed_query = parser.parse_query(
                    query_contents, bind_columns=bind_columns
                )
            except Exception as e:
                raise ValueError(f"Could not parse query from {query_file_path}", e)
            query_label = query_file_path.stem
            queries[label_prefix + query_label] = parsed_query

        return Workload(queries, name=name, root=root)

    def __init__(
        self,
        queries: dict[LabelType, SqlQuery],
        name: str = "",
        root: Optional[pathlib.Path] = None,
    ) -> None:
        super().__init__(queries)
        self._name = name
        self._root = root

        self._sorted_labels = natsort.natsorted(list(self.keys()))
        self._sorted_queries: list[SqlQuery] = []
        self._update_query_order()

        self._label_mapping = util.dicts.invert(self.data)

    @property
    def name(self) -> str:
        """Provides the name of the workload.

        Returns
        -------
        str
            The name or an empty string if no name has been specified.
        """
        return self._name

    def queries(self) -> Sequence[SqlQuery]:
        """Provides all queries in the workload in natural order (according to their labels).

        If the natural order was manually destroyed, e.g. by shuffling, the shuffled order is used.

        Returns
        -------
        Sequence[SqlQuery]
            The queries
        """
        return list(self._sorted_queries)

    def labels(self) -> Sequence[LabelType]:
        """Provides all query labels of the workload in natural order.

        If the natural order was manually destroyed, e.g. by shuffling, the shuffled order is used.

        Returns
        -------
        Sequence[LabelType]
            The labels
        """
        return list(self._sorted_labels)

    def entries(self) -> Sequence[tuple[LabelType, SqlQuery]]:
        """Provides all (label, query) pairs in the workload, in natural order of the query labels.

        If the natural order was manually destroyed, e.g. by shuffling, the shuffled order is used.

        Returns
        -------
        Sequence[tuple[LabelType, SqlQuery]]
            The queries along with their labels
        """
        return list(zip(self._sorted_labels, self._sorted_queries))

    def head(self) -> Optional[tuple[LabelType, SqlQuery]]:
        """Provides the first query in the workload.

        The first query is determined according to the natural order of the query labels by default. If that order was manually
        destroyed, e.g. by shuffling, the shuffled order is used.

        There is no policy to break ties in the order. An arbitrary query can be returned in this case.

        Returns
        -------
        Optional[tuple[LabelType, SqlQuery]]
            The first query, if there is at least one query in the workload. ``None`` otherwise.
        """
        if not self._sorted_labels:
            return None
        return self._sorted_labels[0], self._sorted_queries[0]

    def label_of(self, query: SqlQuery) -> LabelType:
        """Provides the label of the given query.

        Parameters
        ----------
        query : SqlQuery
            The query to check

        Returns
        -------
        LabelType
            The corresponding label

        Raises
        ------
        KeyError
            If the query is not part of the workload
        """
        return self._label_mapping[query]

    def with_labels(self, labels: Iterable[LabelType]) -> Workload[LabelType]:
        """Provides a new workload that contains only the queries with the specified labels.

        Parameters
        ----------
        labels : Iterable[LabelType]
            The labels to include in the new workload

        Returns
        -------
        Workload[LabelType]
            A workload that contains only the queries with the specified labels
        """
        labels = set(labels)
        selected_queries = {
            label: query for label, query in self.data.items() if label in labels
        }
        return Workload(selected_queries, name=self._name, root=self._root)

    def first(self, n: int) -> Workload[LabelType]:
        """Provides the first `n` queries of the workload, according to the natural order of the query labels.

        If there are less than `n` queries in the workload, all queries will be returned. Similar to other methods that rely
        on some sort of ordering of the queries, if the natural order has been manually broken due to shuffling, the shuffled
        order is used instead.

        Parameters
        ----------
        n : int
            The number of queries that should be returned

        Returns
        -------
        Workload[LabelType]
            A workload consisting of the first `n` queries of the current workload
        """
        first_n_labels = self._sorted_labels[:n]
        sub_workload = {label: self.data[label] for label in first_n_labels}
        return Workload(sub_workload, self._name, self._root)

    def last(self, n: int) -> Workload[LabelType]:
        """Provides the last `n` queries of the workload, according to the natural order of the query labels.

        If there are less than `n` queries in the workload, all queries will be returned. Similar to other methods that rely
        on some sort of ordering of the queries, if the natural order has been manually broken due to shuffling, the shuffled
        order is used instead.

        Parameters
        ----------
        n : int
            The number of queries that should be returned

        Returns
        -------
        Workload[LabelType]
            A workload consisting of the last `n` queries of the current workload
        """
        last_n_labels = self._sorted_labels[-n:]
        sub_workload = {label: self.data[label] for label in last_n_labels}
        return Workload(sub_workload, self._name, self._root)

    def pick_random(self, n: int) -> Workload[LabelType]:
        """Constructs a new workload, consisting of randomly selected queries from this workload.

        The new workload will once again be ordered according to the natural ordering of the labels.

        Parameters
        ----------
        n : int
            The number of queries to choose. If there are less queries in the workload, all will be selected.

        Returns
        -------
        Workload[LabelType]
            A workload consisting of `n` unique random queries from this workload
        """
        n = min(n, len(self._sorted_queries))
        selected_labels = random.sample(self._sorted_labels, n)
        sub_workload = {label: self.data[label] for label in selected_labels}
        return Workload(sub_workload, self._name, self._root)

    def with_prefix(self, label_prefix: LabelType) -> Workload[LabelType]:
        """Filters the workload for all queries that have a lablel starting with a specific prefix.

        This method requires that all label instances provide a `startswith` method (as is the case for simple string
        labels). Most significantly, this means that integer-based indexing does not work with for the prefix-based filter.
        The *See Also* section provides some means to mitigate this problem.

        Parameters
        ----------
        label_prefix : LabelType
            The prefix to filter for

        Returns
        -------
        Workload[LabelType]
            All queries of this workload that have a label with a matching prefix. Queries will be sorted according to the
            natural order of their labels again.

        Raises
        ------
        ValueError
            If the prefix type does not provide a `startswith` method.

        See Also
        --------
        relabel - to change the labels into a type that provides `startswith`
        filter_by - to perform a custom prefix check for other types
        """
        if "startswith" not in dir(label_prefix):
            raise ValueError("label_prefix must have startswith() method")
        prefix_queries = {
            label: query
            for label, query in self.data.items()
            if label.startswith(label_prefix)
        }
        return Workload(prefix_queries, name=self._name, root=self._root)

    def filter_by(
        self, predicate: Callable[[LabelType, SqlQuery], bool]
    ) -> Workload[LabelType]:
        """Provides all queries from the workload that match a specific predicate.

        Parameters
        ----------
        predicate : Callable[[LabelType, SqlQuery], bool]
            The filter condition. All queries that pass the check are included in the new workload. The filter predicate
            receives the label and the query for each query in the input

        Returns
        -------
        Workload[LabelType]
            All queries that passed the filter condition check. Queries will be sorted according to the natural order of their
            labels again.
        """
        matching_queries = {
            label: query
            for label, query in self.data.items()
            if predicate(label, query)
        }
        return Workload(matching_queries, name=self._name, root=self._root)

    def relabel(
        self, label_provider: Callable[[LabelType, SqlQuery], NewLabelType]
    ) -> Workload[NewLabelType]:
        """Constructs a new workload, leaving the queries intact but replacing the labels.

        The new workload will ordered according to the natural order of the new labels.

        Parameters
        ----------
        label_provider : Callable[[LabelType, SqlQuery], NewLabelType]
            Replacement method that maps all old labels to the new label values. This method has to provide unique labels. If
            that is not the case, conflicts will be resolved but in an arbitrary way. The replacement receives the old label
            as well as the query as input and produces the new label value.

        Returns
        -------
        Workload[NewLabelType]
            All queries of the current workload, but with new labels
        """
        relabeled_queries = {
            label_provider(current_label, query): query
            for current_label, query in self.data.items()
        }
        return Workload(relabeled_queries, self._name, self._root)

    def shuffle(self) -> Workload[LabelType]:
        """Randomly changes the order of the queries in the workload.

        Returns
        -------
        Workload[LabelType]
            All queries of the current workload, but with the queries in a different order
        """
        shuffled_workload = Workload(self.data, self._name, self._root)
        shuffled_workload._sorted_labels = random.sample(
            self._sorted_labels, k=len(self)
        )
        shuffled_workload._update_query_order()
        return shuffled_workload

    def ordered(self) -> Workload[LabelType]:
        """Enforces the natural ordering of the queries according to their labels.

        Returns
        -------
        Workload[LabelType]
            All queries of the current workload, but in their natural order.
        """
        return Workload(self.data, self._name, self._root)

    def _update_query_order(self) -> None:
        """Enforces that the order of the queries matches the order of the labels."""
        self._sorted_queries = [self.data[label] for label in self._sorted_labels]

    def __add__(self, other: Workload[LabelType]) -> Workload[LabelType]:
        if not isinstance(other, Workload):
            raise TypeError("Can only add workloads together")
        return Workload(
            other.data | self.data, name=self._name, root=self._root
        )  # retain own labels in case of conflict

    def __sub__(self, other: Workload[LabelType]) -> Workload[LabelType]:
        if not isinstance(other, Workload) and isinstance(other, Iterable):
            labels_to_remove = set(other)
            reduced_workload = {
                label: query
                for label, query in self.data.items()
                if label not in labels_to_remove
            }
            return Workload(reduced_workload, name=self._name, root=self._root)
        elif not isinstance(other, Workload):
            raise TypeError("Expected workload or labels to subtract")
        return Workload(
            util.dicts.difference(self.data, other.data),
            name=self._name,
            root=self._root,
        )

    def __and__(self, other: Workload[LabelType]) -> Workload[LabelType]:
        if not isinstance(other, Workload) and isinstance(other, Iterable):
            labels_to_include = set(other)
            reduced_workload = {
                label: query
                for label, query in self.data.items()
                if label in labels_to_include
            }
            return Workload(reduced_workload, name=self._name, root=self._root)
        elif not isinstance(other, Workload):
            raise TypeError("Expected workload or labels to compute union")
        return Workload(
            util.dicts.intersection(self.data, other.data),
            name=self._name,
            root=self._root,
        )

    def __or__(self, other: Workload[LabelType]) -> Workload[LabelType]:
        if not isinstance(other, Workload):
            raise TypeError("Can only compute union of workloads")
        return Workload(
            other.data | self.data, name=self._name, root=self._root
        )  # retain own labels in case of conflict

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self._name:
            return f"Workload: {self._name} ({len(self)} queries)"
        elif self._root:
            return f"Workload: {self._root.stem} ({len(self)} queries)"
        else:
            return f"Workload: {len(self)} queries"


def read_workload(
    path: str,
    name: str = "",
    *,
    query_file_pattern: str = "*.sql",
    recurse_subdirectories: bool = False,
    query_label_prefix: str = "",
    file_encoding: str = "utf-8",
    bind_columns: bool = True,
) -> Workload[str]:
    """Loads a workload consisting of multiple different files, potentially scattered in multiple directories

    The main advantage of this method over using `Workload.read` directly is the support for recursive directory layouts: it
    can traverse subdirectories relative to the specified root and collect all workload files in a recursive manner. If
    subdirectories are used, their names will be used as prefixes to the query label, which is still inferred from the query
    file name.

    Parameters
    ----------
    path : str
        The root directory containing the workload files. Each query is expected to be stored in its own file.
    name : str, optional
        The name of the workload, by default ""
    query_file_pattern : str, optional
        A glob pattern that all query files have to match. All files that match the pattern are assumed to be valid query
        files. Defaults to ``"*.sql"``
    recurse_subdirectories : bool, optional
        Whether query files in subdirectories should be read as well. Defaults to ``False``, which emulates the behaviour of
        `Workload.read`
    query_label_prefix : str, optional
        A global prefix that should be added to all labels, no matter their placement in subdirectories. Defaults to an empty
        string.
    file_encoding : str, optional
        The encoding of the query files. All files must share a common encoding. Defaults to UTF-8

    Returns
    -------
    Workload[str]
        The workload
    """
    base_dir_workload = Workload.read(
        path,
        name=name,
        query_file_pattern=query_file_pattern,
        label_prefix=query_label_prefix,
        file_encoding=file_encoding,
        bind_columns=bind_columns,
    )
    if not recurse_subdirectories:
        return base_dir_workload

    merged_queries = dict(base_dir_workload.data)
    root_dir = pathlib.Path(path)
    for subdir in root_dir.iterdir():
        if not subdir.is_dir():
            continue
        subdir_prefix = (
            (query_label_prefix + "/")
            if query_label_prefix and not query_label_prefix.endswith("/")
            else query_label_prefix
        )
        subdir_prefix += subdir.stem + "/"
        subdir_workload = read_workload(
            str(subdir),
            query_file_pattern=query_file_pattern,
            recurse_subdirectories=True,
            query_label_prefix=subdir_prefix,
            bind_columns=bind_columns,
        )
        merged_queries |= subdir_workload.data
    return Workload(merged_queries, name, root_dir)


def read_batch_workload(
    filename: str, name: str = "", *, file_encoding: str = "utf-8"
) -> Workload[int]:
    """Loads a workload consisting of multiple queries from a single file.

    The input file has to contain one valid SQL query per line. While empty lines are skipped, any non-SQL line will
    raise an Error.

    The workload will have numeric labels: the query in the first line will have label 1, the second one label 2 and
    so on.

    Parameters
    ----------
    filename : str
        The file to load. The extension does not matter, as long as it contains plain text and each query is placed on a single
        and separate line.
    name : str, optional
        The name of the workload. If omitted, this defaults to the file name.
    file_encoding : str, optional
        The encoding of the workload file. Defaults to UTF-8

    Returns
    -------
    Workload[int]
        The workload
    """
    filepath = pathlib.Path(filename)
    name = name if name else filepath.stem
    with open(filename, "r", encoding=file_encoding) as query_file:
        raw_queries = query_file.readlines()
        parsed_queries = [parser.parse_query(raw) for raw in raw_queries if raw]
        return generate_workload(parsed_queries, name=name, workload_root=filepath)


def read_csv_workload(
    filename: str,
    name: str = "",
    *,
    query_column: str = "query",
    label_column: Optional[str] = None,
    file_encoding: str = "utf-8",
    pd_args: Optional[dict] = None,
) -> Workload[str] | Workload[int]:
    """Loads a workload consisting of queries from a CSV column.

    All queries are expected to be contained in the same column and each query is expected to be put onto its own row.

    The column containing the actual queries can be configured via the `query_column` parameter. Likewise, the
    CSV file can already provide query labels in the `label_column` column. If this parameter is omitted, labels will
    be inferred based on the row number.

    Parameters
    ----------
    filename : str
        The name of the CSV file to read. The extension does not matter, as long as the file can be read by the pandas CSV
        parser. The parser can receive additional arguments via the `pd_args` parameter.
    name : str, optional
        The name of the workload. If omitted, this defaults to the file name.
    query_column : str, optional
        The CSV column that contains the workload queries. All rows of that column will be read, by default "query"
    label_column : Optional[str], optional
        The column containing the query labels. Each will receive a label from the `label_column` of the same row. If omitted,
        labels will be inferred based on the row number.
    file_encoding : str, optional
        The encoding of the CSV file. Defaults to UTF-8.
    pd_args : Optional[dict]
        Additional arguments to customize the behaviour of the `pandas.read_csv` method. They will be forwarded as-is. Consult
        the documentation of this method for more details on the allowed parameters and their functionality.

    Returns
    -------
    Workload[str] | Workload[int]
        The workload. It has string labels if `label_column` was provided, or numerical labels otherwise.

    See Also
    ---------
    pandas.read_csv
    """
    filepath = pathlib.Path(filename)
    name = name if name else filepath.stem
    columns = [query_column] + [label_column] if label_column else []

    if pd_args is not None:
        # Prepare the pd_args to not overwrite any of our custom parameters
        pd_args = dict(pd_args)
        pd_args.pop("usecols", None)
        pd_args.pop("converters", None)
        pd_args.pop("encoding", None)

    workload_df = pd.read_csv(
        filename,
        usecols=columns,
        converters={query_column: parser.parse_query},
        encoding=file_encoding,
        **pd_args,
    )

    queries = workload_df[query_column].tolist()
    if label_column:
        labels = workload_df[label_column].tolist()
        label_provider = dict(zip(queries, labels))
    else:
        label_provider = None

    return generate_workload(
        queries, name=name, labels=label_provider, workload_root=filepath
    )


def generate_workload(
    queries: Iterable[SqlQuery],
    *,
    name: str = "",
    labels: Optional[dict[SqlQuery, LabelType]] = None,
    workload_root: Optional[pathlib.Path] = None,
) -> Workload[LabelType]:
    """Wraps a number of queries in a workload object.

    The queries can receive optional labels, and will receive numerical labels according to their position in the `queries`
    iterable if no explicit labels are provided (counting from 1).

    The workload will be named according to the optional `name` parameter. If this fails, the name will be inferred from the
    optional `workload_root`. If this fails as well, an empty name will be used.

    Parameters
    ----------
    queries : Iterable[SqlQuery]
        The queries that should form the workload. This is only enumerated a single time, hence the iterable can "spent" its
        items.
    name : str, optional
        The name of the workload, by default ""
    labels : Optional[dict[SqlQuery, LabelType]], optional
        The labels of the workload queries. Defaults to ``None``, in which case numerical labels will be used. In the first
        case the label type is inferred from the dictionary values. In the second case, it will be `int`.
    workload_root : Optional[pathlib.Path], optional
        The directory or file that originally contained the workload queries. Defaults to ``None`` if this is not known or not
        appropriate (e.g. for workloads that are read from a remote source)

    Returns
    -------
    Workload[LabelType]
        The workload
    """
    name = name if name else (workload_root.stem if workload_root else "")
    if not labels:
        labels: dict[SqlQuery, int] = {
            query: idx + 1 for idx, query in enumerate(queries)
        }
    workload_contents = util.dicts.invert(labels)
    return Workload(workload_contents, name, workload_root)


def _assert_workload_loaded(workload: Workload[LabelType], expected_dir: str) -> None:
    """Ensures that workload queries have been read successfully. The expected directory is used for error messages."""
    if not workload:
        wdir = os.getcwd()
        raise ValueError(
            f"Could not load {workload.name} workload. This is likely due to a disparity between workload "
            "location and current value of the workloads_base_dir setting. Make sure to point that variable to "
            f"the correct path. Your current working directory is '{wdir}' and the expected workload directory "
            f"is '{expected_dir}'"
        )


def job(file_encoding: str = "utf-8") -> Workload[str]:
    """Reads the Join Order Benchmark, as shipped with the PostBOUND repository.

    Queries will be read from the JOB directory relative to `workloads_base_dir`. The expected layout is:
    ``<workloads_base_dir>/JOB-Queries/<queries>``. Labels are inferred from the file names, i.e. queries are accessible as
    ``1a``, ``8c``, ``33b`` and so on.

    Parameters
    ----------
    file_encoding : str, optional
        The encoding of the query files, by default UTF-8.

    Returns
    -------
    Workload[str]
        The workload

    Raises
    ------
    ValueError
        If the workload could not be loaded from the expected location

    References
    ----------

    .. Viktor Leis et al.: "How Good Are Query Optimizers, Really?" (Proc. VLDB Endow. 9, 3 (2015))
    """
    job_dir = os.path.join(workloads_base_dir, "JOB-Queries")
    # JOB only uses aliases column references, so no need for explicit binding
    job_workload = Workload.read(
        job_dir, name="JOB", file_encoding=file_encoding, bind_columns=False
    )
    _assert_workload_loaded(job_workload, job_dir)
    return job_workload


def job_light(file_encoding: str = "utf-8") -> Workload[str]:
    """Reads the JOB-light benchmark, as shipped with the PostBOUND repository.

    Queries will be read from the JOB directory relative to `workloads_base_dir`. The expected layout is:
    ``<workloads_base_dir>/JOB-Light-Queries/<queries>``. Labels are inferred from the file names, i.e. queries are
    accessible as ``1``, ``2``, ``3`` and so on.

    Parameters
    ----------
    file_encoding : str, optional
        The encoding of the query files, by default UTF-8.

    Returns
    -------
    Workload[str]
        The workload

    Raises
    ------
    ValueError
        If the workload could not be loaded from the expected location

    References
    ----------

    .. Andreas Kipf et al.: "Learned Cardinalities: Estimating Correlated Joinswith Deep Learning" (CIDR'2019)
    """
    job_light_dir = os.path.join(workloads_base_dir, "JOB-light-Queries")
    # JOB-light only uses aliases column references, so no need for explicit binding
    job_light_workload = Workload.read(
        job_light_dir, name="JOB-light", file_encoding=file_encoding, bind_columns=False
    )
    _assert_workload_loaded(job_light_workload, job_light_dir)
    return job_light_workload


def ssb(
    file_encoding: str = "utf-8", *, bind_columns: Optional[bool] = None
) -> Workload[str]:
    """Reads the Star Schema Benchmark, as shipped with the PostBOUND repository.

    Queries will be read from the SSB directory relative to `workloads_base_dir`. The expected layout is:
    ``<workloads_base_dir>/SSB-Queries/<queries>``. Labels are inferred from the file names, i.e. queries are accessible as
    ``q1-1``, ``q4-3``, etc.

    Parameters
    ----------
    file_encoding : str, optional
        The encoding of the query files, by default UTF-8.

    Returns
    -------
    Workload[str]
        The workload

    Raises
    ------
    ValueError
        If the workload could not be loaded from the expected location

    References
    ----------

    .. Patrick E. O'Neil et al.: "The Star Schema Benchmark and Augmented Fact Table Indexing." (TPCTC'2009)
    """
    bind_columns = (
        bind_columns
        if bind_columns is not None
        else not DatabasePool.get_instance().empty()
    )
    ssb_dir = os.path.join(workloads_base_dir, "SSB-Queries")
    ssb_workload = Workload.read(
        ssb_dir, name="SSB", file_encoding=file_encoding, bind_columns=bind_columns
    )
    _assert_workload_loaded(ssb_workload, f"{workloads_base_dir}/SSB-Queries")
    return ssb_workload


def _fetch_stack_queries(path: str) -> None:
    """Utility method to load the Stack queries if they are not available.

    Parameters
    ----------
    path : str
        The path to the Stack-Queries directory
    """
    current_path = os.getcwd()
    os.chdir(os.path.join(workloads_base_dir, "Stack-Queries"))
    if "q1" in os.listdir():
        os.chdir(current_path)
        return

    os.system("./setup.sh")
    os.chdir(current_path)


def stack(
    file_encoding: str = "utf-8",
    *,
    bind_columns: Optional[bool] = None,
    fetch: bool = True,
) -> Workload[str]:
    """Reads the Stack Benchmark, as shipped with the PostBOUND repository.

    Queries will be read from the Stack directory relative to `workloads_base_dir`. The expected layout is:
    ``<workloads_base_dir>/Stack-Queries/<sub-directories>/<queries>``. Alternatively, all queries can be contained in
    the ``Stack-Queries`` directory directly. Labels are inferred from the file names, i.e. queries are accessible as
    ``q1/q1-001``, ``q4/q4-100``, etc. Notice that there are also some queries with entirely "random" file names, such as
    ``q16/fc8f97968b9fce81df4011c8175eada15541abe0``.

    Parameters
    ----------
    file_encoding : str, optional
        The encoding of the query files, by default UTF-8.
    bind_columns : Optional[bool], optional
        Whether to bind columns in the queries. If omitted, this is determined based on the current database connection.
    fetch : bool, optional
        Whether the workload queries should be fetched if they are not loaded already. This requires a working internet
        connection.

    Returns
    -------
    Workload[str]
        The workload.

    Raises
    ------
    ValueError
        If the workload could not be loaded from the expected location

    Notes
    -----
    Notice that the Stack Benchmark is much much larger than the Join Order Benchmark or the Star Schema Benchmark.
    Therefore, the benchmark queries are not put in version control directly, instead they have to be manually loaded.
    You can use the `fetch` parameter to load the queries automatically, if it appears that they are not available.
    See the documentation in the workload directory for details on the loading logic.

    References
    ----------

    .. Ryan Marcus et al.: "Bao: Making Learned Query Optimization Practical." (SIGMOD'2021)
    """
    bind_columns = (
        bind_columns
        if bind_columns is not None
        else not DatabasePool.get_instance().empty()
    )
    stack_dir = os.path.join(workloads_base_dir, "Stack-Queries")
    if fetch:
        _fetch_stack_queries(stack_dir)

    stack_workload = read_workload(
        stack_dir,
        "Stack",
        recurse_subdirectories=True,
        file_encoding=file_encoding,
        bind_columns=bind_columns,
    )
    _assert_workload_loaded(stack_workload, stack_dir)
    return stack_workload


def stats(file_encoding: str = "utf-8") -> Workload[str]:
    """Reads the Stats Benchmark, as shipped with the PostBOUND repository.

    Queries will be read from the Stats directory relative to `workload_base_dir`. The expected layout is:
    ``<workloads_base_dir>/Stats-CEB/queries/<queries>``. Labels are inferred from the file names, i.e. queries are accessible
    as ``q-1``, ``q-2``, etc. This labelling is custom to PostBOUND, since the original workload did not include any meaningful
    labels or candidates for such.

    Parameters
    ----------
    file_encoding : str, optional
        The encoding of the query files, by default UTF-8.

    Returns
    -------
    Workload[str]
        The workload

    Raises
    ------
    ValueError
        If the workload could not be loaded from the expected location

    References
    ----------

    .. Yuxing Han et al.: Cardinality Estimation in DBMS: A Comprehensive Benchmark Evaluation (Proc. VLDB Endow. 15, 4 (2022))
    """
    stats_dir = os.path.join(workloads_base_dir, "Stats-CEB", "queries")
    stats_workload = Workload.read(
        stats_dir, name="Stats", file_encoding=file_encoding, bind_columns=False
    )
    _assert_workload_loaded(stats_workload, stats_dir)
    return stats_workload
