__all__ = [
    "read_array_float",
    "read_array_int",
    "write_array_float",
    "write_array_int",
]
from pathlib import Path

import numpy as np
from arraystubs import Arr, Arr2


def read_array_int[I: np.integer](
    name: Path | str,
    skip: int = 0,
    *,
    dtype: type[I] = np.intc,
) -> Arr[tuple[int, ...], I]:
    return np.loadtxt(name, skiprows=skip, dtype=dtype)


def read_array_float[F: np.floating](
    name: Path | str,
    skip: int = 0,
    *,
    dtype: type[F] = np.float64,
) -> Arr[tuple[int, ...], F]:
    return np.loadtxt(name, skiprows=skip, dtype=dtype)


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
