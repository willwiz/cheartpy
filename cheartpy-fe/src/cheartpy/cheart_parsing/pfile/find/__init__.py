from typing import TYPE_CHECKING

from cheartpy.cheart_parsing.pfile._regex import OutputPath, get_output_path

if TYPE_CHECKING:
    from pathlib import Path


def find_output_dir(pfile: Path) -> Path | None:
    text = pfile.read_text()
    for line in text.splitlines():
        match get_output_path(line):
            case OutputPath(path):
                return path
            case None:
                continue
    return None
