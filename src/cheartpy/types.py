__all__ = ["Arr", "f64", "i32", "char", "Vec", "Mat", "MatV"]
import numpy as np
from typing import Any, TextIO

Arr = np.ndarray
f64 = np.dtype[np.float64]
i32 = np.dtype[np.int32]
char = np.dtype[np.str_]
bool_ = np.dtype[np.bool_]


type Vec[T: (i32, f64, char, bool_)] = np.ndarray[tuple[int], T]
type Mat[T: (i32, f64, char, bool_)] = np.ndarray[tuple[int, int], T]
type MatV[T: (i32, f64, char, bool_)] = np.ndarray[tuple[int, int, int], T]
