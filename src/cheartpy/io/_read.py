import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeIs, overload

import numpy as np
from pytools.arrays import SupportsDType

if TYPE_CHECKING:
    from pytools.arrays import A1, A2, DType


type ReadFormat = Literal["D", "X", "T", "B", "Time", "Raw"]


def fix_ch_sfx[T: (Path | str)](prefix: T, suffix: str = "_FE.") -> T:
    _prefix = str(prefix)
    for i in range(len(suffix), 0, -1):
        if _prefix.endswith(suffix[:i]):
            _prefix = _prefix + suffix[i:]
            break
    else:
        _prefix = _prefix + suffix
    return type(prefix)(_prefix)


def check_for_meshes(*names: str, home: Path | None = None, bc: bool = True) -> bool:
    home = home or Path()
    sfx = ["X", "T", "B"] if bc else ["X", "T"]
    meshes = [w for name in names for w in [f"{fix_ch_sfx(name)}{s}" for s in sfx]]
    return all((home / s).is_file() for s in meshes)


def is_binary(filename: Path | str) -> bool:
    filename = Path(filename)
    try:
        # or codecs.open on Python <= 2.5
        # or io.open on Python > 2.5 and <= 2.7
        with filename.open("r") as f:
            _ = [int(i) for i in next(f).strip().split()]
    except ValueError:
        try:
            # or codecs.open on Python <= 2.5
            # or io.open on Python > 2.5 and <= 2.7
            with filename.open("rb") as f:
                _ = [float(i) for i in next(f).strip().split()]
        except ValueError:
            return True
        else:
            return False
    else:
        return False


def chread_d_utf[F: np.number](file: Path | str, *, dtype: DType[F] = np.float64) -> A2[F]:
    return np.loadtxt(file, skiprows=1, dtype=dtype, ndmin=2)


def chread_d_bin[F: np.number](file: Path | str, *, dtype: DType[F] = np.float64) -> A2[F]:
    with Path(file).open("rb") as f:
        nnodes = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
        arr = np.zeros((nnodes, dim), dtype=dtype)
        for i in range(nnodes):
            for j in range(dim):
                bite = f.read(8)
                if not bite:
                    msg = "binary buffer ran out before indicated range"
                    raise BufferError(msg)
                arr[i, j] = struct.unpack("d", bite)[0]
    return arr


def chread_d[F: np.number](file: Path | str, *, dtype: DType[F] = np.float64) -> A2[F]:
    if is_binary(file):
        return chread_d_bin(file, dtype=dtype)
    return chread_d_utf(file, dtype=dtype)


def chread_data[F: np.floating](file: Path | str, *, dtype: DType[F] = np.float64) -> A2[F]:
    file = Path(file)
    if is_binary(file):
        return chread_d_bin(file, dtype=dtype)
    with file.open("r") as f:
        first_line = f.readline().strip("\n")
    matched = re.match(r"\s*(\d+)\s+(\d+)\s*$", first_line)
    if matched:
        return chread_d_utf(file, dtype=dtype)
    return np.loadtxt(file, dtype=dtype)


def chread_t_utf[I: np.integer](file: Path | str, *, dtype: DType[I] = np.intp) -> A2[I]:
    return np.loadtxt(file, skiprows=1, dtype=dtype, ndmin=2)


def chread_b_utf[I: np.integer](file: Path | str, *, dtype: DType[I] = np.intp) -> A2[I]:
    return np.loadtxt(file, skiprows=1, dtype=dtype, ndmin=2)


def chread_time_utf[F: np.floating, I: np.integer](
    file: Path | str, *, dtype: DType[I] = np.intp, ftype: DType[F] = np.float64
) -> A1[F]:
    data = np.loadtxt(file, dtype=[("index", dtype), ("value", ftype)], skiprows=1)
    shape = (data["index"].max() + 1,)
    time = np.zeros(shape, dtype=ftype)
    time[data["index"]] = data["value"]
    return time


def chread_header_utf(file: Path | str) -> tuple[int, int]:
    with Path(file).open("r") as f:
        items = next(f).strip().split()
        nelem = int(items[0])
        nnode = int(items[1])
    return nelem, nnode


def _is_floating_dtype(dtype: DType[np.number]) -> TypeIs[np.dtype[np.floating]]:
    # 1. Normalize the DType union into a unified type object
    match dtype:
        # Case A: Input is already an np.dtype object
        case np.dtype() as dt if issubclass(dt.type, np.floating):
            return True
        # Case C: Input is a duck-typed object carrying a .dtype attribute
        case SupportsDType():
            return issubclass(dtype.dtype.type, np.floating)
        # Case B: Input is a type object itself (e.g., np.float64, float)
        case type() as t if issubclass(t, np.floating):
            return True
        case _:
            return False


def _is_integer_dtype(dtype: DType[np.number]) -> TypeIs[np.dtype[np.integer]]:
    # 1. Normalize the DType union into a unified type object
    match dtype:
        # Case A: Input is already an np.dtype object
        case np.dtype() as dt if issubclass(dt.type, np.integer):
            return True
        # Case C: Input is a duck-typed object carrying a .dtype attribute
        case SupportsDType():
            return issubclass(dtype.dtype.type, np.integer)
        # Case B: Input is a type object itself (e.g., np.int32, int)
        case type() as t if issubclass(t, np.integer):
            return True
        case _:
            return False


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
def chread(
    file: Path | str,
    *,
    fmt: ReadFormat = "D",
    dtype: DType[np.number] | None = None,
    ftype: DType[np.number] | None = None,
) -> A1[np.number] | A2[np.number] | A2[np.void]:
    """Read data from a file in Cheart related formats.

    Parameters
    ----------
    file : Path | str
        The file path to read the data from.
    fmt : ReadFormat, optional
        The format of the data in the file. Options are:
        - "D": Cheart data format (default)
        - "X": Cheart extended data format
        - "T": Cheart topology format
        - "B": Cheart boundary format
        - "Time": Cheart time format
        - "Raw": Raw text format
    dtype : DType[np.number] | None, optional
        The data type to use for reading the numbers. If None, defaults to np.float64 for "D" and
        "X", np.intp for "T" and "B", and np.float64 for "Time" and "Raw".
    ftype : DType[np.number] | None, optional
        The floating-point data type to use for reading the time values in "Time" format. If None,
        defaults to np.float64.

    Returns
    -------
    A1[np.number] | A2[np.number] | A2[np.void]
        The data read from the file, as a 1D or 2D array of numbers, or a structured array for
        "Time" format.

    """
    match fmt:
        case "D" | "X":
            return chread_d(file, dtype=dtype or np.float64)
        case "T":
            t = dtype or np.intp
            if not _is_integer_dtype(t):
                msg = f"Please follow chread overload signature. Expect integer dtype, got {t}"
                raise TypeError(msg)
            return chread_t_utf(file, dtype=t)
        case "B":
            t = dtype or np.intp
            if not _is_integer_dtype(t):
                msg = f"Please follow chread overload signature. Expect integer dtype, got {t}"
                raise TypeError(msg)
            return chread_b_utf(file, dtype=t)
        case "Time":
            f = ftype or np.float64
            if not _is_floating_dtype(f):
                msg = f"Please follow chread overload signature. Expect floating dtype, got {f}"
                raise TypeError(msg)
            d = dtype or np.intp
            if not _is_integer_dtype(d):
                msg = f"Please follow chread overload signature. Expect integer dtype, got {d}"
                raise TypeError(msg)
            return chread_time_utf(file, dtype=d, ftype=f)
        case "Raw":
            return np.loadtxt(file, dtype=dtype)
