"""Contains various utilities that did not fit any other category."""
from __future__ import annotations

from typing import Any


def _wrap_version(v: Any) -> Version:
    """Transforms any object into a Version instance if it is not already."""
    return v if isinstance(v, Version) else Version(v)


class Version:
    """Version instances represent versioning information and ensure that comparison operations work as expected.

    For example, Version instances can be created for strings such as "14.6" or "1.3.1" and ensure that 14.6 > 1.3.1
    """

    def __init__(self, ver: str | int | list[str] | list[int]) -> None:
        if isinstance(ver, int):
            self._version = [ver]
        elif isinstance(ver, str):
            self._version = [int(v) for v in ver.split(".")]
        elif isinstance(ver, list) and ver:
            self._version = [int(v) for v in ver]
        else:
            raise ValueError(f"Unknown version string: '{ver}'")

    def formatted(self, *, prefix: str = "", suffix: str = "", separator: str = "."):
        return prefix + separator.join(str(v) for v in self._version) + suffix

    def __eq__(self, __o: object) -> bool:
        try:
            other = _wrap_version(__o)
            if not len(self) == len(other):
                return False
            for i in range(len(self)):
                if not self._version[i] == other._version[i]:
                    return False
            return True
        except ValueError:
            return False

    def __ge__(self, __o: object) -> bool:
        return not self < __o

    def __gt__(self, __o: object) -> bool:
        return not self <= __o

    def __le__(self, __o: object) -> bool:
        other = _wrap_version(__o)
        for comp in zip(self._version, other._version):
            own_version, other_version = comp
            if own_version < other_version:
                return True
            if other_version < own_version:
                return False
        return len(self) <= len(other)

    def __lt__(self, __o: object) -> bool:
        other = _wrap_version(__o)
        for comp in zip(self._version, other._version):
            own_version, other_version = comp
            if own_version < other_version:
                return True
            if other_version < own_version:
                return False
        return len(self) < len(other)

    def __len__(self) -> int:
        return len(self._version)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return "v" + ".".join(str(v) for v in self._version)
