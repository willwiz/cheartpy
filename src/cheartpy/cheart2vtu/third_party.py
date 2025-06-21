# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

from pathlib import Path

__all__ = ["compress_vtu"]

from typing import TYPE_CHECKING

import meshio
from pytools.logging.api import BLogger

if TYPE_CHECKING:
    from pytools.logging.trait import ILogger


def compress_vtu(name: Path | str, log: ILogger | None = None) -> None:
    """Read the name of a file and compresses it in vtu format."""
    if log is None:
        log = BLogger("DEBUG")
    log.debug(f"File size before: {Path(name).stat().st_size / 1024**2:.2f} MB")
    mesh = meshio.read(name, file_format="vtu")
    meshio.vtu.write(name, mesh, binary=True, compression="zlib")
    log.debug(f"File size after: {Path(name).stat().st_size / 1024**2:.2f} MB")
