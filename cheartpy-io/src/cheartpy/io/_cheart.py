import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeGuard, TypeIs, overload

import numpy as np
from pytools.arrays import SupportsDType

if TYPE_CHECKING:
    from pytools.arrays import A1, A2, Arr, DType

"""
CHeart Read Array functions
"""

__all__ = [
    "check_for_meshes",
    "chread_b_utf",
    "chread_d",
    "chread_d_bin",
    "chread_d_utf",
    "chread_header_utf",
    "chread_t_utf",
    "chwrite_d_binary",
    "chwrite_d_utf",
    "chwrite_iarr_utf",
    "chwrite_t_utf",
    "chwrite_time_utf",
    "fix_ch_sfx",
    "is_binary",
]

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
    meshes = [w for name in names for w in [f"{name}_FE.{s}" for s in sfx]]
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
def chread[T: np.floating](
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


"""
CHeart Write Array functions
"""


def chwrite_d_binary[T: np.number, S: tuple[int, ...]](file: Path | str, data: Arr[S, T]) -> None:
    match data.shape:
        case (int(size),):
            dim = 1
        case int(size), int(dim):
            ...
        case _:
            msg = "Data must be 1D or 2D array"
            raise ValueError(msg)
    with Path(file).open("wb") as f:
        f.write(struct.pack("i", size))
        f.write(struct.pack("i", dim))
        for i in data:
            f.writelines(struct.pack("d", j) for j in i)


def chwrite_d_utf[T: np.number, S: tuple[int, ...]](file: Path | str, data: Arr[S, T]) -> None:
    match data.shape:
        case (int(),):
            ne = data.size
            nn = 1
        case int(), int():
            ne, nn = data.shape
        case _:
            msg = "Data must be 1D or 2D array"
            raise ValueError(msg)
    if np.issubdtype(data.dtype, np.integer):
        fmt = "%16d"
    elif np.issubdtype(data.dtype, np.floating):
        fmt = "%24.16e"
    else:
        msg = f"Unsupported data type: {data.dtype}"
        raise TypeError(msg)
    np.savetxt(
        file,
        data,
        fmt=fmt,
        delimiter=" ",
        newline="\n",
        header=f"{ne:12d}{nn:12d}",
        comments="",  # Avoids the default '# ' comment prefix
    )


def chwrite_i_utf[T: np.integer, S: tuple[int, ...]](file: Path | str, data: Arr[S, T]) -> None:
    match data.shape:
        case (int(),):
            ne = data.size
            nn = 1
        case int(), int():
            ne, nn = data.shape
        case _:
            msg = "Data must be 1D or 2D array"
            raise ValueError(msg)
    np.savetxt(
        file,
        data,
        delimiter=" ",
        fmt="%g",
        newline="\n",
        header=f"{ne:12d}{nn:12d}",
        comments="",  # Avoids the default '# ' comment prefix
    )


def chwrite_list_utf[T: np.number](
    file: Path | str, data: A1[T], *, dtype: DType[T] = np.float64
) -> None:
    """Write a 1D array of numbers to a file.

    Useful for node assignments.

    Parameters
    ----------
    file : Path | str
        The file path to write the data to.
    data : A1[T]
        The 1D array of numbers to write.
    dtype : DType[T], optional
        The data type to use for writing the numbers (default: np.float64).

    Raises
    ------
    ValueError
        If the input data is not a 1D array.

    """
    nn = data.shape[0]
    np.savetxt(
        file,
        np.asarray(data[:, np.newaxis], dtype),
        fmt="%16d" if np.issubdtype(dtype, np.integer) else "%24.16e",
        delimiter=" ",
        newline="\n",
        header=f"{nn:12d}",
        comments="",  # Avoids the default '# ' comment prefix
    )


def chwrite_t_utf[T: np.integer](file: Path | str, data: A2[T], nn: int | None = None) -> None:
    ne = len(data)
    nn = int(data.max()) if nn is None else nn
    if data.ndim != 2:  # noqa: PLR2004
        msg = "Topology must be 2D array of integers"
        raise ValueError(msg)
    with Path(file).open("w") as f:
        f.write(f"{ne:12d}")
        f.write(f"{nn:12d}\n")
        for i in data:
            f.writelines(f"{j:>12d}" for j in i)
            f.write("\n")


def chwrite_iarr_utf[T: np.integer](file: Path | str, data: A2[T]) -> None:
    dim = data.shape
    with Path(file).open("w") as f:
        f.write(f"{dim[0]:12d}\n")
        for i in data:
            f.writelines(f"{j:>12d}" for j in i)
            f.write("\n")


def chwrite_str_utf[T: np.str_](file: Path | str, data: A2[T]) -> None:
    with Path(file).open("w") as f:
        f.write(f"{data.shape[0]:>12}")
        f.write(f"{data.shape[1]:>12}\n")
        for i in data:
            f.writelines(f"{j:>12}" for j in i)
            f.write("\n")


def chwrite_time_utf[F: np.floating](file: Path | str, data: A1[F]) -> None:
    with Path(file).open("w") as f:
        f.write(f"{len(data):>12}\n")
        f.writelines(f"{i:>12}{v:>24.16g}\n" for i, v in enumerate(data, start=1))


@overload
def chwrite[T: np.number](file: Path | str, data: A1[T], *, binary: bool = ...) -> None: ...
@overload
def chwrite[T: np.number](file: Path | str, data: A2[T], *, binary: bool = ...) -> None: ...
@overload
def chwrite[T: np.number](
    file: Path | str, data: A1[T], *, fmt: Literal["D"], binary: bool = ...
) -> None: ...
@overload
def chwrite[T: np.number](
    file: Path | str, data: A2[T], *, fmt: Literal["D"], binary: bool = ...
) -> None: ...
@overload
def chwrite[T: np.number](file: Path | str, data: A2[T], *, fmt: Literal["X"]) -> None: ...
@overload
def chwrite[T: np.integer](file: Path | str, data: A2[T], *, fmt: Literal["T"]) -> None: ...
@overload
def chwrite[T: np.integer](file: Path | str, data: A2[T], *, fmt: Literal["B"]) -> None: ...
@overload
def chwrite[F: np.floating](file: Path | str, data: A1[F], *, fmt: Literal["Time"]) -> None: ...
@overload
def chwrite[T: np.number](file: Path | str, data: A2[T], *, fmt: Literal["Raw"]) -> None: ...


def _is_1d_array[T: np.generic](
    arr: Arr[tuple[int] | tuple[int, int], np.number], dtype: DType[T]
) -> TypeGuard[A1[T]]:
    return arr.ndim == 1 and np.issubdtype(arr.dtype, dtype)


def _is_2d_array[T: np.generic](
    arr: Arr[tuple[int] | tuple[int, int], np.number], dtype: DType[T]
) -> TypeGuard[A2[T]]:
    return arr.ndim != 1 and np.issubdtype(arr.dtype, dtype)


def chwrite(
    file: Path | str,
    data: Arr[tuple[int] | tuple[int, int], np.number],
    *,
    fmt: ReadFormat = "D",
    binary: bool = False,
) -> None:
    match fmt:
        case "D" | "X":
            if binary:
                chwrite_d_binary(file, data)
            else:
                chwrite_d_utf(file, data)
        case "T":
            if not _is_2d_array(data, np.integer):
                msg = (
                    "Please follow chwrite overload signature. Expected 2D integer array,"
                    f"got {data.dtype} with shape {data.shape}"
                )
                raise ValueError(msg)
            chwrite_t_utf(file, data)
        case "B":
            if not _is_2d_array(data, np.integer):
                msg = (
                    "Please follow chwrite overload signature. Expected 2D integer array,"
                    f"got {data.dtype} with shape {data.shape}"
                )
                raise ValueError(msg)
            chwrite_iarr_utf(file, data)
        case "Time":
            if not _is_1d_array(data, np.floating):
                msg = (
                    "Please follow chwrite overload signature. Expected 1D floating array,"
                    f"got {data.dtype} with shape {data.shape}"
                )
                raise ValueError(msg)
            chwrite_time_utf(file, data)
        case "Raw":
            np.savetxt(file, data)
