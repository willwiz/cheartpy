# type: ignore
# meshio does not have properly implemented stubs, strict type checking will fail
__all__ = ["compress_vtu"]
import os
import meshio


def compress_vtu(name: str, verbose: bool = False) -> None:
    """
    Reads the name of a file and compresses it in vtu format
    """
    if verbose:
        size = os.stat(name).st_size
        print("File size before: {:.2f} MB".format(size / 1024**2))
    mesh = meshio._helpers.read(name, file_format="vtu")
    meshio.vtu.write(name, mesh, binary=True, compression="zlib")
    if verbose:
        size = os.stat(name).st_size
        print("File size after: {:.2f} MB".format(size / 1024**2))
