from pathlib import Path
from typing import TYPE_CHECKING, Unpack

from pytools.logging import get_logger

from ._regex import Unparsed, ValueLine, parse_line
from .find import find_output_dir

if TYPE_CHECKING:
    from ._types import PFileParserKwargs


def parse_pfile_api(file: Path | str, **kwargs: Unpack[PFileParserKwargs]) -> int:
    file = Path(file)
    log = kwargs.get("logger") or get_logger()
    log.info(f"Parsing {file.name}...")
    if not file.exists():
        log.error(f"File {file} does not exist")
        return 1
    if file.suffix != ".P":
        msg = f"Expected a .P file, got {file.suffix}"
        log.error(msg)
        return 1
    text = file.read_text()
    items = dict(enumerate(filter(None, [parse_line(line) for line in text.splitlines()])))
    objects = {
        k: v
        for k, v in items.items()
        if not isinstance(v, ValueLine) and not isinstance(v, Unparsed)
    }
    unparsed = {k: v for k, v in items.items() if isinstance(v, Unparsed)}
    log.info(objects)
    log.info(unparsed)
    path = find_output_dir(file)
    print(f"Output path: {path}")
    return 0
