from __future__ import annotations

from dataclasses import dataclass

from ..util import jsondict
from ..workloads import Workload


@dataclass
class BenchmarkConfig:
    workload: Workload


class BenchmarkState:
    def refresh(self) -> BenchmarkState:
        pass


class Benchmark:
    def start(self) -> BenchmarkState:
        pass

    def stop(self) -> BenchmarkState:
        pass

    def resume(self) -> BenchmarkState:
        pass


class BenchmarkConfigurator:
    def add_stage(self) -> BenchmarkConfigurator:
        pass

    def done(self) -> BenchmarkConfigurator:
        """Exits the current benchmark stage and resumes configuration of the containing stage.

        If there is no higher-level stage, resumes configuration of the main benchmark settings.
        """
        pass

    def build(self) -> Benchmark:
        pass


def setup() -> BenchmarkConfigurator:
    pass


def serialize_config(benchmark: Benchmark) -> jsondict:
    pass


def load(config: str | jsondict) -> Benchmark:
    pass
