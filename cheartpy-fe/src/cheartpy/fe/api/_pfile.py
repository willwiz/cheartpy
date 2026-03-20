from typing import TYPE_CHECKING

from cheartpy.fe.impl import PFile

if TYPE_CHECKING:
    from pathlib import Path


def create_pfile(header: str = "", output_dir: Path | None = None) -> PFile:
    return PFile(header, output_dir=output_dir)
