import numpy as np
from typing import Any

__all__ = ["Arr", "Any", "f64", "int_t", "char", "Vec", "Mat", "MatV", "V2", "T3"]
Arr = np.ndarray
f64 = np.dtype[np.float64]
int_t = np.dtype[np.int_]
char = np.dtype[np.str_]
bool_ = np.dtype[np.bool_]
type Vec[T: (int_t, f64, char, bool_)] = np.ndarray[tuple[int], T]
type Mat[T: (int_t, f64, char, bool_)] = np.ndarray[tuple[int, int], T]
type MatV[T: (int_t, f64, char, bool_)] = np.ndarray[tuple[int, int, int], T]
type V2[T: (float, int)] = tuple[T, T]
type T3[T: (float, int)] = tuple[T, T, T]
