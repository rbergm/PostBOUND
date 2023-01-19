
import pathlib
from typing import Dict, List, Tuple, Union

from transform import mosp


class Workload:
    def __init__(self, root_dir: str, query_file_pattern: str = "*.sql", *, name: str = ""):
        self._queries: Dict[str, str] = {}
        self._root = pathlib.Path(root_dir)
        self._query_pattern = query_file_pattern
        self._name = name

        for query_file_path in self._root.glob(query_file_pattern):
            with open(query_file_path, "r", encoding="utf-8") as query_file:
                raw_contents = query_file.readlines()
            query_contents = " ".join([line for line in raw_contents if not line.startswith("--")])
            parsed_query = mosp.MospQuery.parse(query_contents)
            query_label = query_file_path.stem
            self._queries[query_label] = parsed_query

    def queries(self) -> List[mosp.MospQuery]:
        return list(self._queries.values())

    def items(self) -> List[Tuple[str, mosp.MospQuery]]:
        return [(label, query) for label, query in self._queries.items()]

    def contents(self) -> Dict[str, mosp.MospQuery]:
        return dict(self._queries)

    def labels(self) -> List[str]:
        return list(self._queries.keys())

    def __getitem__(self, key: str) -> Union[None, mosp.MospQuery]:
        return self._queries.get(key, None)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        name = self._name if self._name else self._root.stem
        return f"Workload: {name} ({len(self._queries)} queries)"


def job() -> Workload:
    return Workload("../workloads/JOB-Queries/implicit", name="JOB")


def ssb() -> Workload:
    return Workload("../workloads/SSB-Queries", name="SSB")
