
import abc

class OptimizationSettings(abc.ABC):
    pass

class UESOptimizationSettings(OptimizationSettings):
    pass


def fetch(key: str) -> OptimizationSettings:
    pass
