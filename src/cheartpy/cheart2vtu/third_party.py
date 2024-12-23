# type: ignore
# meshio does not have properly implemented stubs, strict type checking will fail
__all__ = ["compress_vtu"]
import os
from ..tools.basiclogging import BLogger, ILogger, LogLevel
import meshio


def compress_vtu(name: str, LOG: ILogger = BLogger("NULL")) -> None:
    """
    Reads the name of a file and compresses it in vtu format
    """
    if LOG.level > LogLevel.INFO:
        size = os.stat(name).st_size
        LOG.debug("File size before: {:.2f} MB".format(size / 1024**2))
    mesh = meshio._helpers.read(name, file_format="vtu")
    meshio.vtu.write(name, mesh, binary=True, compression="zlib")
    if LOG.level > LogLevel.INFO:
        size = os.stat(name).st_size
        LOG.debug("File size after: {:.2f} MB".format(size / 1024**2))
