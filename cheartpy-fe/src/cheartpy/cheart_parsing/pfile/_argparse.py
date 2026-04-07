import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError
from pytools.result import Err, Ok, Result

from ._types import PFileParserArgs, PFileParserKwargs

if TYPE_CHECKING:
    from collections.abc import Sequence

_parser = argparse.ArgumentParser()
_parser.add_argument("file", nargs="+", type=Path)


class _Args(BaseModel):
    file: list[Path]


def parse_args(
    args: Sequence[str] | None = None,
) -> Result[tuple[PFileParserArgs, PFileParserKwargs]]:
    try:
        parsed_args = _Args(**vars(_parser.parse_args(args)))
    except ValidationError as e:
        return Err(e)
    _args = PFileParserArgs(file=parsed_args.file)
    _kwargs = PFileParserKwargs()
    return Ok((_args, _kwargs))
