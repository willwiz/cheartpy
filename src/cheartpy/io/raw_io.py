from __future__ import annotations

from pathlib import Path

import numpy as np
from arraystubs import Arr, Arr2


def read_array_int(name: Path | str, skip: int = 0) -> Arr[tuple[int, ...], np.intc]:
    return np.loadtxt(name, skiprows=skip, dtype=np.intc)


def read_array_float(name: Path | str, skip: int = 0) -> Arr[tuple[int, ...], np.float64]:
    return np.loadtxt(name, skiprows=skip, dtype=np.float64)


def write_array_int[T: np.integer](name: Path | str, arr: Arr2[T]) -> None:
    with Path(name).open("w") as f:
        for i in arr:
            f.writelines(f"{j:12d}" for j in i)
            f.write("\n")


def write_array_float[T: np.floating](name: Path | str, arr: Arr2[T]) -> None:
    with Path(name).open("w") as f:
        for i in arr:
            f.writelines(f"{j:>24.12E}" for j in i)
            f.write("\n")
