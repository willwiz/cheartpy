import numpy as np
from ..var_types import *


def read_array_int(name: str, skip: int = 0) -> Arr[Any, i32]:
    return np.loadtxt(name, skiprows=skip, dtype=int)


def read_array_float(name: str, skip: int = 0) -> Arr[Any, f64]:
    return np.loadtxt(name, skiprows=skip, dtype=float)


def write_array_int(name: str, arr: Arr[Any, i32]) -> None:
    with open(name, "w") as f:
        for i in arr:
            for j in i:
                f.write("{:12d}".format(j))
            f.write("\n")


def write_array_float(name: str, arr: Arr[Any, f64]) -> None:
    with open(name, "w") as f:
        for i in arr:
            for j in i:
                f.write("{:>24.12E}".format(j))
            f.write("\n")
