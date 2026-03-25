from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Self, get_args

import pandas as pd

from . import util
from .parser import parse_query
from .util import jsondict

TrainingMetrics = jsondict

TrainingFeature = Literal[
    "query", "runtime_ms", "query_plan", "estimated_cost", "cardinality"
]


class TrainingSpec:
    def __init__(self, features: Iterable[TrainingFeature]) -> None:
        self._features: list[TrainingFeature] = list(features)
        self._feature_set = frozenset(self._features)

    @property
    def features(self) -> Sequence[TrainingFeature]:
        return self._features

    @property
    def feature_set(self) -> frozenset[TrainingFeature]:
        return self._feature_set

    def provides(self, feature: TrainingFeature | Iterable[TrainingFeature]) -> bool:
        feature: set[TrainingFeature] = set(util.enlist(feature))
        return feature.issubset(self._feature_set)

    def requires(self, feature: TrainingFeature | Iterable[TrainingFeature]) -> bool:
        return self.provides(feature)

    def satisfies(self, other: TrainingSpec) -> SpecViolations:
        missing = other._feature_set - self._feature_set
        return SpecViolations(missing)

    def __iter__(self):
        return iter(self.features)

    def __hash__(self) -> int:
        return hash(self._feature_set)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, TrainingSpec) and self._feature_set == other._feature_set
        )

    def __repr__(self) -> str:
        return f"TrainingSpec({self.features})"

    def __str__(self) -> str:
        features = ", ".join(self.features)
        return f"TrainingSpec({features})"


class SpecViolations:
    def __init__(self, missing_features: frozenset[TrainingFeature]) -> None:
        self.missing_features = missing_features

    def __bool__(self) -> bool:
        return not bool(self.missing_features)

    def __repr__(self) -> str:
        return f"SpecViolations(missing_features={self.missing_features})"

    def __str__(self) -> str:
        if not self.missing_features:
            return "SpecViolations(no violations)"
        missing = ", ".join(self.missing_features)
        return f"SpecViolations(missing_features={missing})"


@dataclass
class TrainingData:
    @staticmethod
    def from_df(
        df: pd.DataFrame | Path | str, *, source: Optional[Path | str] = None
    ) -> TrainingData:
        if isinstance(df, (str, Path)):
            source = Path(df)
            df = util.read_df(source)
        if isinstance(source, str):
            source = Path(source)

        detected_features: list[TrainingFeature] = []
        available_features = set(get_args(TrainingFeature))
        for col in df.columns:
            if col not in available_features:
                continue
            detected_features.append(col)

        feature_map: dict[TrainingFeature, str] = {
            feat: feat for feat in detected_features
        }
        query_col = feature_map.get("query")
        if query_col:
            df[query_col] = df[query_col].map(parse_query)

        return TrainingData(df, source=source, feature_map=feature_map)

    @staticmethod
    def merge(
        datasets: Iterable[TrainingData], *, according_to: TrainingSpec
    ) -> TrainingData:
        datasets = list(datasets)
        merged, *rest = datasets
        for ds in rest:
            merged = merged.merge_with(ds)
        return merged.conform_to(according_to)

    def __init__(
        self,
        samples: pd.DataFrame,
        *,
        source: Optional[Path] = None,
        feature_map: dict[TrainingFeature, str],
    ) -> None:
        self._source = source
        self._feature_map = feature_map
        self._samples = samples
        self._spec = TrainingSpec(self._feature_map.keys())

    @property
    def source(self) -> Optional[Path]:
        return self._source

    @property
    def feature_map(self) -> Mapping[TrainingFeature, str]:
        return self._feature_map

    @property
    def samples(self) -> pd.DataFrame:
        return self._samples

    @property
    def spec(self) -> TrainingSpec:
        return self._spec

    def provides(self, feature: TrainingFeature) -> bool:
        return self._spec.provides(feature)

    def satisfies(self, spec: TrainingSpec) -> bool:
        return bool(self._spec.satisfies(spec))

    def conform_to(
        self, features: Iterable[TrainingFeature] | TrainingSpec
    ) -> TrainingData:
        spec = (
            features if isinstance(features, TrainingSpec) else TrainingSpec(features)
        )
        if not self._spec.satisfies(spec):
            raise ValueError("Requested spec is not compatible with the training data")
        reduced_spec: dict[TrainingFeature, str] = {
            feature: col
            for feature, col in self._feature_map.items()
            if spec.requires(feature)
        }
        return TrainingData(
            self._samples, source=self._source, feature_map=reduced_spec
        )

    def as_df(self, requested_spec: TrainingSpec | None = None) -> pd.DataFrame:
        if requested_spec is None:
            target_cols = self._feature_map.values()
        elif not self._spec.satisfies(requested_spec):
            raise ValueError("Requested spec is not compatible with the training data")
            target_cols = [self.feature_map[feature] for feature in requested_spec]
        else:
            target_cols = requested_spec.features
        return self._samples[target_cols]

    def merge_with(
        self,
        other: TrainingData,
    ) -> TrainingData:
        violations = other._spec.satisfies(self._spec)
        if violations:
            raise ValueError(
                "Cannot merge the given training data: "
                f"the other dataset is missing features {violations.missing_features}"
            )

        other_samples = other.as_df(self._spec)
        merged_samples = pd.concat([self._samples, other_samples])
        return TrainingData(merged_samples, source=None, feature_map=self._feature_map)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)

    def __getitem__(self, idx) -> list:
        sample = self._samples.iloc[idx]
        remapped = sample[self._feature_map.values()]
        return list(remapped)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        features = ", ".join(self._feature_map.keys())
        source = self._source if self._source else "intermediate"
        return f"TrainingData({source}, features=[{features}])"


class TrainingDataRepository:
    def __init__(self) -> None:
        self._datasets: list[TrainingData] = []

    def register(self, samples: TrainingData) -> Self:
        self._datasets.append(samples)
        return self

    def retrieve_first(self, spec: TrainingSpec) -> Optional[TrainingData]:
        dataset = next((ds for ds in self._datasets if ds.satisfies(spec)), None)
        if dataset is None:
            return None
        return dataset.conform_to(spec)

    def retrieve_all(self, spec: TrainingSpec) -> Sequence[TrainingData]:
        return [ds for ds in self._datasets if ds.satisfies(spec)]

    def retrieve_merged(self, spec: TrainingSpec) -> Optional[TrainingData]:
        matching = self.retrieve_all(spec)
        if not matching:
            return None
        elif len(matching) == 1:
            return matching[0]
        else:
            return TrainingData.merge(matching, according_to=spec)
