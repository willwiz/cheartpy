import argparse
from pathlib import Path
from typing import TYPE_CHECKING, get_args

from pydantic import BaseModel, ValidationError
from pytools.logging import LogLevel
from pytools.result import Err, Ok, Result

from ._traits import VarCompAPIArgs, VarCompAPIKwargs

if TYPE_CHECKING:
    from collections.abc import Sequence

parser = argparse.ArgumentParser(
    description="""
    Compare the values of two arrays imported from  file
    """,
)
parser.add_argument(
    "--folder",
    "-f",
    type=Path,
    nargs=1,
    action="extend",
    help="OPTIONAL: Relative path to data directory. Checks the current directory if not found.",
)
parser.add_argument(
    "--log-level",
    choices=get_args(LogLevel.__value__),
    type=str.upper,
    default="INFO",
    help="OPTIONAL: logging level",
)
parser.add_argument(
    "var",
    type=str,
    nargs="+",
    metavar=("var1, var2"),
    help=(
        "get name or prefix of first files, careful of shell expansion of wildcards. "
        "If only `var1` is provided, `var2 = var1` is assumed."
    ),
)


class VarCompModel(BaseModel):
    var: list[str]
    folder: list[Path] | None
    log_level: LogLevel


_NVARS = 2


def set_variable_prefixes(args: VarCompModel) -> Result[tuple[str, str]]:
    match args.var:
        case [var1] if args.folder and len(args.folder) == _NVARS:
            var2 = var1
        case [var1, var2]: ...  # fmt: skip
        case [var1]:
            msg = f"Only one var = `{var1}` given with 1 or 0 folder. Pointless comparison"
            return Err(ValueError(msg))
        case _:
            msg = "Error: Too many variables provided. Please provide at most two variables."
            return Err(ValueError(msg))
    return Ok((var1, var2))


def set_root_directories(folders: Sequence[Path] | None) -> Result[tuple[Path, Path]]:
    match folders:
        case [] | None:
            f1, f2 = Path.cwd(), Path.cwd()
        case [f1]:
            f2 = f1
        case [f1, f2]: ...  # fmt: skip
        case _:
            msg = "Error: Too many folders provided. Please provide at most two folders."
            return Err(ValueError(msg))
    return Ok((f1, f2))


def parse_cmdline_args(
    args: Sequence[str] | None = None,
) -> Result[tuple[VarCompAPIArgs, VarCompAPIKwargs]]:
    try:
        parsed_args = VarCompModel(**vars(parser.parse_args(args)))
    except ValidationError as e:
        return Err(e)
    match set_root_directories(parsed_args.folder):
        case Ok((root_1, root_2)): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match set_variable_prefixes(parsed_args):
        case Ok((var_1, var_2)): ...  # fmt: skip
        case Err(e):
            return Err(e)
    _args = VarCompAPIArgs(var_1=var_1, var_2=var_2)
    _kwargs = VarCompAPIKwargs(root_1=root_1, root_2=root_2, log_level=parsed_args.log_level)
    return Ok((_args, _kwargs))
