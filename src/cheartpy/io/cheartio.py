import struct
from cheartpy.types import i32, f64, Arr, char
import numpy as np

"""
CHeart Read Array functions
"""


def is_binary(filename):
    try:
        # or codecs.open on Python <= 2.5
        # or io.open on Python > 2.5 and <= 2.7
        with open(filename) as f:
            line = f.readline().strip().split()
            _ = [int(i) for i in line]
        return False
    except:
        try:
            # or codecs.open on Python <= 2.5
            # or io.open on Python > 2.5 and <= 2.7
            with open(filename) as f:
                line = f.readline().strip().split()
                _ = [float(i) for i in line]
            return False
        except:
            return True


def CHRead_d_utf(file: str) -> Arr[tuple[int, int], f64]:
    return np.loadtxt(file, skiprows=1, dtype=float, ndmin=2)


def CHRead_d_bin(file: str) -> Arr[tuple[int, int], f64]:
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


def CHRead_d(file: str) -> Arr[tuple[int, int], f64]:
    if is_binary(file):
        return CHRead_d_bin(file)
    return CHRead_d_utf(file)


def CHRead_t_utf(file: str) -> Arr[tuple[int, int], i32]:
    return np.loadtxt(file, skiprows=1, dtype=np.int32, ndmin=2)


def CHRead_header_utf(file: str) -> tuple[int, int]:
    with open(file, "r") as f:
        items = f.readline().strip().split()
        nelem = int(items[0])
        nnode = int(items[1])
    return nelem, nnode


def CHRead_b_utf(file: str) -> Arr[tuple[int, int], i32]:
    return np.loadtxt(file, skiprows=1, dtype=int, ndmin=2)


"""
CHeart Write Array functions
"""


def CHWrite_d_utf(file: str, data: Arr[tuple[int, int], f64]) -> None:
    dim = data.shape
    with open(file, "w") as f:
        f.write("{:12d}".format(dim[0]))
        f.write("{:12d}\n".format(dim[1]))
        for i in data:
            for j in i:
                f.write("{:>24.12E}".format(j))
            f.write("\n")
    return


def CHWrite_d_binary(file: str, arr: Arr[tuple[int, int], f64]) -> None:
    dim = arr.shape
    with open(file, "wb") as f:
        f.write(struct.pack("i", dim[0]))
        f.write(struct.pack("i", dim[1]))
        for i in arr:
            for j in i:
                f.write(struct.pack("d", j))
    return


def CHWrite_t_utf(file: str, data: Arr[tuple[int, int], i32], ne: int, nn: int) -> None:
    with open(file, "w") as f:
        f.write(f"{ne:12d}")
        f.write(f"{nn:12d}\n")
        for i in data:
            for j in i:
                f.write(f"{j:>12d}")
            f.write("\n")
    return


def CHWrite_iarr_utf(file: str, data: Arr[tuple[int, int], i32]) -> None:
    dim = data.shape
    with open(file, "w") as f:
        f.write(f"{dim[0]:12d}\n")
        for i in data:
            for j in i:
                f.write(f"{j:>12d}")
            f.write("\n")
    return


def CHWrite_Str_utf(file: str, data: Arr[tuple[int, int], char]) -> None:
    with open(file, "w") as f:
        f.write("{:>12}".format(data.shape[0]))
        f.write("{:>12}\n".format(data.shape[1]))
        for i in data:
            for j in i:
                f.write("{:>12}".format(j))
            f.write("\n")
    return
