import argparse
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from pytools.result import Err, Ok, Result
from pytools.typing import is_type

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
    "--suffix",
    "-s",
    dest="suffix",
    type=str,
    default="Quad",
    help="OPTIONAL: output file will have [tag] appended before index numbers and extension",
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
    help="file path to linear mesh",
)
interp_parser.add_argument(
    "--quad",
    "-q",
    type=str,
    required=True,
    help="file path to quadratic mesh",
)
interp_parser.add_argument(
    "vars",
    nargs="+",
    help="names to files/variables.",
    type=str,
)


class InterpArgs(TypedDict, total=True):
    lin: str
    quad: str
    vars: list[str]


class InterpKwargs(TypedDict, total=False):
    suffix: str
    input_dir: Path
    ext: str


_ARG_TYPES: dict[str, type] = {
    "lin": str,
    "quad": str,
    "suffix": str,
    "input_dir": Path,
    "ext": Literal["D", "D.gz"],
}


def parser_interp_args(args: Mapping[str, object]) -> Result[tuple[InterpArgs, InterpKwargs]]:
    for k, kind in _ARG_TYPES.items():
        if (v := args.get(k)) and not is_type(v, kind):
            msg = f"Argument '{k}' must be of type {kind.__name__}"
            return Err(TypeError(msg))
        if v is None:
            return Err(ValueError(f"Missing required argument '{k}'")) if v is None else Ok(None)
    vs = args.get("vars")
    if not vs:
        msg = "At least one variable must be specified for interpolation"
        return Err(ValueError(msg))
    if not (isinstance(vs, list) and all(isinstance(v, str) for v in vs)):
        msg = "All variable names must be strings"
        return Err(TypeError(msg))
    _args_dict: InterpArgs = {
        "lin": args["lin"],
        "quad": args["quad"],
        "vars": args["vars"],
    }
    _kwargs_dict: InterpKwargs = {
        "suffix": args["suffix"],
        "input_dir": args["input_dir"],
        "ext": args["ext"],
    }
    return Ok((_args_dict, _kwargs_dict))


def get_interp_args(args: list[str] | None = None) -> tuple[InterpArgs, InterpKwargs]:
    namespace = interp_parser.parse_args(args)
    match parser_interp_args(vars(namespace)):
        case Ok(result):
            return result
        case Err(e):
            print(f"Error parsing arguments: {e}")
            interp_parser.print_help()
            raise SystemExit(1)
