from __future__ import annotations

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
    "fix_suffix",
    "is_binary",
]
import struct
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from arraystubs import Arr, Arr2


"""
CHeart Read Array functions
"""


def is_binary(filename: Path | str) -> bool:
    filename = Path(filename)
    try:
        # or codecs.open on Python <= 2.5
        # or io.open on Python > 2.5 and <= 2.7
        with filename.open("r") as f:
            line = f.readline().strip().split()
            _ = [int(i) for i in line]
    except ValueError:
        try:
            # or codecs.open on Python <= 2.5
            # or io.open on Python > 2.5 and <= 2.7
            with filename.open("rb") as f:
                line = f.readline().strip().split()
                _ = [float(i) for i in line]
        except ValueError:
            return True
        else:
            return False
    else:
        return False


def fix_suffix(prefix: str, suffix: str = "_FE.") -> str:
    for i in range(len(suffix), 0, -1):
        if prefix.endswith(suffix[:i]):
            return prefix + suffix[i:]
    return prefix + suffix


def check_for_meshes(*names: str, bc: bool = True) -> bool:
    sfx = ["X", "T", "B"] if bc else ["X", "T"]
    meshes = [w for name in names for w in [f"{name}_FE.{s}" for s in sfx]]
    return all(Path(s).exists() for s in meshes)


def chread_d_utf(file: Path | str) -> Arr2[np.float64]:
    return np.loadtxt(file, skiprows=1, dtype=np.float64, ndmin=2)


def chread_d_bin(file: Path | str) -> Arr2[np.float64]:
    with Path(file).open("rb") as f:
        nnodes = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
        arr = np.zeros((nnodes, dim), dtype=np.float64)
        for i in range(nnodes):
            for j in range(dim):
                bite = f.read(8)
                if not bite:
                    msg = "binary buffer ran out before indicated range"
                    raise BufferError(msg)
                arr[i, j] = struct.unpack("d", bite)[0]
    return arr


def chread_d(file: Path | str) -> Arr2[np.float64]:
    if is_binary(file):
        return chread_d_bin(file)
    return chread_d_utf(file)


def chread_t_utf(file: Path | str) -> Arr2[np.intc]:
    return np.loadtxt(file, skiprows=1, dtype=np.intc, ndmin=2)


def chread_header_utf(file: Path | str) -> tuple[int, int]:
    with Path(file).open("r") as f:
        items = f.readline().strip().split()
        nelem = int(items[0])
        nnode = int(items[1])
    return nelem, nnode


def chread_b_utf(file: Path | str) -> Arr2[np.intc]:
    return np.loadtxt(file, skiprows=1, dtype=np.intc, ndmin=2)


"""
CHeart Write Array functions
"""


def chwrite_d_utf[T: np.floating, S: tuple[int, ...]](file: Path | str, data: Arr[S, T]) -> None:
    match data.shape:
        case (int(),):
            ne = data.size
            nn = 1
        case int(), int():
            ne, nn = data.shape
        case _:
            msg = "Data must be 1D or 2D array"
            raise ValueError(msg)
        # case _:
        #     raise ValueError("Data must be 1D or 2D array")
    # if data.ndim == 1:
    #     data = cast(Mat[f64], data[:, None])
    # elif data.ndim > 2:
    # ne, nn = data.shape
    np.savetxt(
        file,
        data,
        fmt="%24.16e",
        delimiter=" ",
        newline="\n",
        header=f"{ne:12d}{nn:12d}",
        comments="",  # Avoids the default '# ' comment prefix
    )


def chwrite_d_binary[T: np.floating](file: Path | str, arr: Arr2[T]) -> None:
    dim = arr.shape
    with Path(file).open("wb") as f:
        f.write(struct.pack("i", dim[0]))
        f.write(struct.pack("i", dim[1]))
        for i in arr:
            f.writelines(struct.pack("d", j) for j in i)


def chwrite_t_utf[T: np.integer](file: Path | str, data: Arr2[T], nn: int | None = None) -> None:
    ne = len(data)
    nn = data.max() if nn is None else nn
    if data.ndim != 2:  # noqa: PLR2004
        msg = "Topology must be 2D array of integers"
        raise ValueError(msg)
    with Path(file).open("w") as f:
        f.write(f"{ne:12d}")
        f.write(f"{nn:12d}\n")
        for i in data:
            f.writelines(f"{j:>12d}" for j in i)
            f.write("\n")


def chwrite_iarr_utf[T: np.integer](file: Path | str, data: Arr2[T]) -> None:
    dim = data.shape
    with Path(file).open("w") as f:
        f.write(f"{dim[0]:12d}\n")
        for i in data:
            f.writelines(f"{j:>12d}" for j in i)
            f.write("\n")


def chwrite_str_utf[T: np.str_](file: Path | str, data: Arr2[T]) -> None:
    with Path(file).open("w") as f:
        f.write(f"{data.shape[0]:>12}")
        f.write(f"{data.shape[1]:>12}\n")
        for i in data:
            f.writelines(f"{j:>12}" for j in i)
            f.write("\n")
