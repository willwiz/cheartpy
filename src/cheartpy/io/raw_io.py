from pathlib import Path

import numpy as np
from arraystubs import Arr2


def read_array_int[T: np.integer](name: str, skip: int = 0) -> Arr2[T]:
    return np.loadtxt(name, skiprows=skip, dtype=int)


def read_array_float[T: np.floating](name: str, skip: int = 0) -> Arr2[T]:
    return np.loadtxt(name, skiprows=skip, dtype=float)


def write_array_int[T: np.integer](name: str, arr: Arr2[T]) -> None:
    with Path(name).open("w") as f:
        for i in arr:
            f.writelines(f"{j:12d}" for j in i)
            f.write("\n")


def write_array_float[T: np.floating](name: str, arr: Arr2[T]) -> None:
    with Path(name).open("w") as f:
        for i in arr:
            f.writelines(f"{j:>24.12E}" for j in i)
            f.write("\n")
