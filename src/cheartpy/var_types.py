__all__ = ["AUTO", "Arr", "Any", "f64", "i32", "char", "Vec", "Mat", "MatV", "V2", "V3"]
import numpy as np
from typing import Any, Final

Arr = np.ndarray
f64 = np.dtype[np.float64]
i32 = np.dtype[np.int32]
char = np.dtype[np.str_]
bool_ = np.dtype[np.bool_]

type Vec[T: (i32, f64, char, bool_)] = np.ndarray[tuple[int], T]
type Mat[T: (i32, f64, char, bool_)] = np.ndarray[tuple[int, int], T]
type MatV[T: (i32, f64, char, bool_)] = np.ndarray[tuple[int, int, int], T]
type V2[T: (float, int)] = tuple[T, T]
type V3[T: (float, int)] = tuple[T, T, T]


class _Auto(object):
    def __init__(self) -> None:
        pass

    def __eq__(self, value: object) -> bool:
        if isinstance(value, self.__class__):
            return True
        return False

    def __bool__(self) -> bool:
        return True


AUTO: Final[_Auto] = _Auto()
