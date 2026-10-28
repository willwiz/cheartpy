import struct
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeGuard, overload

import numpy as np

if TYPE_CHECKING:
    from pytools.arrays import A1, A2, Arr, DType

    from ._read import ReadFormat


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
