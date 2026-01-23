# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from pathlib import Path
from typing import TypedDict, Unpack

import meshio
from pytools.logging import NLOGGER, ILogger

__all__ = ["compress_vtu"]


class _Kwargs(TypedDict, total=False):
    log: ILogger


def compress_vtu(name: Path | str, **kwargs: Unpack[_Kwargs]) -> None:
    """Read the name of a file and compresses it in vtu format."""
    log = kwargs.get("log", NLOGGER)
    log.debug(f"File size before: {Path(name).stat().st_size / 1024**2:.2f} MB")
    mesh = meshio.read(name, file_format="vtu")
    meshio.vtu.write(name, mesh, binary=True, compression="zlib")
    log.debug(f"File size after: {Path(name).stat().st_size / 1024**2:.2f} MB")
