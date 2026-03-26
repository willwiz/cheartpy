import argparse
from pathlib import Path
from typing import TypedDict

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


def get_interp_args(args: list[str] | None = None) -> tuple[InterpArgs, InterpKwargs]:
    namespace = interp_parser.parse_args(args)
    _args_dict: InterpArgs = {
        "lin": namespace.lin,
        "quad": namespace.quad,
        "vars": namespace.vars,
    }
    _kwargs_dict: InterpKwargs = {
        "suffix": namespace.suffix,
        "input_dir": namespace.input_dir,
        "ext": namespace.ext,
    }
    return _args_dict, _kwargs_dict
