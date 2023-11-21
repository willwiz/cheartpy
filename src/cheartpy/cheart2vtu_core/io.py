import os
import meshio
import struct
import numpy as np
from cheartpy.cheart2vtu_core.data_types import Arr, f64


def read_D_binary(file: str) -> Arr[tuple[int, int], f64]:
    with open(file, mode="rb") as f:
        nnodes = struct.unpack("i", f.read(4))[0]
        dim = struct.unpack("i", f.read(4))[0]
        arr = np.zeros((nnodes, dim))
        for i in range(nnodes):
            for j in range(dim):
                bite = f.read(8)
                if not bite:
                    raise BufferError(
                        "Binary buffer being read ran out before indicated range"
                    )
                arr[i, j] = struct.unpack("d", bite)[0]
    return arr


def read_D_utf(file: str) -> Arr[tuple[int, int], f64]:
    return np.loadtxt(file, skiprows=1, dtype=float)


def read_T_utf(file: str) -> Arr[tuple[int, int], f64]:
    return np.loadtxt(file, skiprows=1, dtype=int)


def compress_vtu(name: str, verbose: bool = False) -> None:
    if verbose:
        size = os.stat(name).st_size
        print("File size before: {:.2f} MB".format(size / 1024**2))
    mesh = meshio._helpers.read(name, file_format="vtu")
    meshio.vtu.write(name, mesh, binary=True, compression="zlib")
    if verbose:
        size = os.stat(name).st_size
        print("File size after: {:.2f} MB".format(size / 1024**2))
