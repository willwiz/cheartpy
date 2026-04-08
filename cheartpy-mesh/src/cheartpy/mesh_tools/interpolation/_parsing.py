import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict
from warnings import warn

from pydantic import BaseModel, ValidationError
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Mapping

interp_parser = argparse.ArgumentParser(
    "interp",
    description="""interpolate data from linear topology to quadratic topology""",
)
interp_parser.add_argument(
    "--folder",
    "-f",
    type=Path,
    default=Path(),
    dest="input_dir",
    help="OPTIONAL: specify a folder for the where the variables are stored. NOT YET IMPLEMENTED",
)
interp_parser.add_argument(
    "--postfix",
    "-p",
    dest="postfix",
    type=str,
    nargs="+",
    default=None,
    help="OPTIONAL: [<postfix>, ...]. Same size as input. Append `_Quad` if not given",
)
interp_parser.add_argument(
    "--ext",
    dest="ext",
    choices=["D", "D.gz"],
    default="D",
    help="OPTIONAL: D file suffix",
)
interp_parser.add_argument(
    "--lin",
    "-l",
    type=str,
    required=True,
    help="REQUIRED: file path to linear mesh",
)
interp_parser.add_argument(
    "--quad",
    "-q",
    type=str,
    required=True,
    help="REQUIRED: file path to quadratic mesh",
)
interp_parser.add_argument(
    "--threads",
    "-n",
    type=int,
    default=1,
    help="OPTIONAL: number of threads to use for interpolation. Default is 1.",
)
interp_parser.add_argument(
    "--overwrite",
    action="store_true",
    help="OPTIONAL: whether to overwrite existing files.",
)
interp_parser.add_argument(
    "vars",
    nargs="+",
    help="names to files/variables.",
    type=str,
)


class ArgsModel(BaseModel):
    lin: str
    quad: str
    vars: Sequence[str]
    postfix: Sequence[str] | None


class KwargsModel(BaseModel):
    input_dir: Path = Path.cwd()
    overwrite: bool = False
    threads: int = 1
    ext: Literal["D", "D.gz"] = "D"


class InterpArgs(TypedDict, total=True):
    lin: str
    quad: str
    vars: Sequence[str]
    postfix: Sequence[str] | None


class InterpKwargs(TypedDict, total=False):
    input_dir: Path
    overwrite: bool
    threads: int
    ext: Literal["D", "D.gz"]


def parser_interp_args(args: Mapping[str, Any]) -> Result[tuple[InterpArgs, InterpKwargs]]:
    try:
        interp_args = ArgsModel(**args)
    except ValidationError as e:
        return Err(e)
    try:
        interp_kwargs = KwargsModel(**args)
    except ValidationError as e:
        return Err(e)
    return Ok((InterpArgs(**interp_args.model_dump()), InterpKwargs(**interp_kwargs.model_dump())))


def get_interp_args(args: list[str] | None = None) -> tuple[InterpArgs, InterpKwargs]:
    namespace = interp_parser.parse_args(args)
    match parser_interp_args(vars(namespace)):
        case Ok(result):
            return result
        case Err(e):
            warn(f"Error parsing arguments: {e}", stacklevel=2)
            interp_parser.print_help()
            raise SystemExit(1)
