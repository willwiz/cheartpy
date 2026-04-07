from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from pytools.logging import ILogger


class PFileParserArgs(TypedDict, total=True):
    file: list[Path]


class PFileParserKwargs(TypedDict, total=False):
    logger: ILogger
