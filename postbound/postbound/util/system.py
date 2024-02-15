"""Provides utilities to access (operating system) related information."""
from __future__ import annotations

import os
from typing import Optional

from postbound.util import proc


def open_files(pid: Optional[int] = None) -> list[str]:
    """Provides all files (e.g. text files and shared objects) opened by the given process/PID.

    Parameters
    ----------
    pid : Optional[int], optional
        The PID of the process to query. Defaults to the current process.

    Returns
    -------
    list[str]
        All opened files
    """
    pid = os.getpid() if pid is None else pid
    res = proc.run_cmd(f"lsof -p {pid}" + "| awk '{print $9}' | grep -E '*.so'", shell=True)
    return res.splitlines()
