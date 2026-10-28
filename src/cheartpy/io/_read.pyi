from pathlib import Path
from typing import Literal, overload

import numpy as np
from pytools.arrays import A2, DType

type ReadFormat = Literal["D", "X", "T", "B", "Time", "Raw"]

def fix_ch_sfx[T: (Path | str)](prefix: T, suffix: str = "_FE.") -> T: ...
def check_for_meshes(*names: str, home: Path | None = None, bc: bool = True) -> bool: ...
def is_binary(filename: Path | str) -> bool: ...
def chread_data[F: np.floating](file: Path | str, *, dtype: DType[F] = np.float64) -> A2[F]: ...
def chread_header_utf(file: Path | str) -> tuple[int, int]: ...
@overload
def chread[T: np.number](file: Path | str, *, dtype: DType[T] = np.float64) -> A2[T]: ...
@overload
def chread[T: np.number](
    file: Path | str, *, fmt: Literal["D"], dtype: DType[T] = np.float64
) -> A2[T]: ...
@overload
def chread[T: np.floating](
    file: Path | str, *, fmt: Literal["X"], dtype: DType[T] = np.float64
) -> A2[T]: ...
@overload
def chread[T: np.integer](
    file: Path | str, *, fmt: Literal["T"], dtype: DType[T] = np.intp
) -> A2[T]: ...
@overload
def chread[T: np.integer](
    file: Path | str, *, fmt: Literal["B"], dtype: DType[T] = np.intp
) -> A2[T]: ...
@overload
def chread[F: np.floating, I: np.integer](
    file: Path | str,
    *,
    fmt: Literal["Time"],
    dtype: DType[I] = np.intp,
    ftype: DType[F] = np.float64,
) -> A2[np.void]: ...
@overload
def chread[T: np.number](
    file: Path | str, *, fmt: Literal["Raw"], dtype: DType[T] = np.float64
) -> A2[T]: ...
