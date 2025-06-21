__all__ = ["T2", "T3", "Any", "Arr", "Mat", "MatV", "Vec", "char", "f64", "int_t"]
from typing import Any

import numpy as np

Arr = np.ndarray
type f64 = np.dtype[np.floating]
type int_t = np.dtype[np.int_]
type char = np.dtype[np.str_]
type bool_ = np.dtype[np.bool_]
type Vec[T: (int_t, f64, char, bool_)] = np.ndarray[tuple[int, ...], T]
type Mat[T: (int_t, f64, char, bool_)] = np.ndarray[tuple[int, ...], T]
type MatV[T: (int_t, f64, char, bool_)] = np.ndarray[tuple[int, ...], T]
type T2[T: (float, int)] = tuple[T, T]
type T3[T: (float, int)] = tuple[T, T, T]
