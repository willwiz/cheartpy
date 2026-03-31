import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

from pydantic import BaseModel, ValidationError
from pytools.logging import LogLevel
from pytools.result import Err, Ok, Result

from ._types import AbaqusAPIArgs, AbaqusAPIKwargs
from ._utils import gather_masks, split_argslist_to_nameddict

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

################################################################################################
# Check if multiprocessing is available


_parser = argparse.ArgumentParser(
    description="""
    Convert Abaqus mesh to Cheart. Main() can be editted for convenience, see example at
    the bottom. Example inputs:

    Default: Exports all elements with default name as mesh_ele_FE.T files.
      python3 abaqus2cheart.py mesh.inp

    With Topology defined as the element Volume:
      python3 abaqus2cheart.py mesh.inp -t Volume

    With Boundaries:
      Surface 1 labeled as 1
      Surfaces 2 3 4 labeled as 2
      Topology as Volume1 and Volume2
      python3 abaqus2cheart.py mesh.inp -t Volume1 Volume2 -b Surface1 1 -b Surface2 Surface3 2

    Mesh is check for errors if topology and boundary as indicated. Extra nodes are not included.

""",
    formatter_class=RawTextHelpFormatter,
)
_parser.add_argument(
    "files",
    nargs="+",
    type=str,
    help="""Name of the .inp file containing the Abaqus mesh. If given after the
    optional arguments -t or -b, -- should be inserted in between to delineate.
    """,
)
_parser.add_argument(
    "-t",
    "--topology",
    required=True,
    type=str,
    nargs="+",
    default=None,
    help="""Define which volume will be used as the topology. If multiple are given,
    they are appended. E.g.,
    --topology Volume1
    --topology Volume1 Volume2 Volume3 ...
    """,
)
_parser.add_argument(
    "-p",
    "--prefix",
    type=str,
    help="""Give the prefix for the output files.""",
)
_parser.add_argument(
    "-b",
    "--boundary",
    type=str,
    action="append",
    nargs="+",
    default=None,
    help="""Set a boundary give the name of the element and label or name, appended
    numerals, and label. E.g.,
    --boundary Surf1 label
    --boundary Surf1 Surf2 ... label
    """,
)
_parser.add_argument(
    "--add-mask",
    type=str,
    action="append",
    nargs="+",
    default=None,
    help="""Export masks with given labels. E.g.,
    --add-mask Surf1 value file_name.ext
    --add-mask Volume1 Volume2 ... SurfN value file_name.ext

    """,
)
_parser.add_argument(
    "--log-level",
    type=str.upper,
    default="INFO",
    choices=get_args(LogLevel),
    help="""Set the log level for the program.""",
)
_parser.add_argument("-c", "--cores", type=int, help="""Enable multiprocessing with n cores""")


class PydanticParser(BaseModel):
    files: Sequence[Path | str]
    topology: Sequence[str]
    boundary: Sequence[Sequence[str]] | None
    add_mask: Sequence[Sequence[str]] | None
    prefix: str | None
    log_level: LogLevel
    cores: int


def parse_api_kwargs(dct: Mapping[str, Any]) -> Result[tuple[AbaqusAPIArgs, AbaqusAPIKwargs]]:
    try:
        parsed_args = PydanticParser(**dct)
    except ValidationError as e:
        return Err(e)
    prefix = parsed_args.prefix or Path(parsed_args.files[0]).stem
    match split_argslist_to_nameddict(parsed_args.boundary):
        case Ok(boundary): ...  # fmt: skip
        case Err(e):
            return Err(e)
    match gather_masks(parsed_args.add_mask):
        case Ok(masks): ...  # fmt: skip
        case Err(e):
            return Err(e)
    args = AbaqusAPIArgs(
        files=parsed_args.files,
    )
    kwargs = AbaqusAPIKwargs(
        topology=parsed_args.topology,
        boundary=boundary,
        masks=masks,
        prefix=prefix,
        log_level=parsed_args.log_level,
        cores=parsed_args.cores,
    )
    return Ok((args, kwargs))


def parse_cmdline_args(args: Sequence[str] | None = None) -> tuple[AbaqusAPIArgs, AbaqusAPIKwargs]:
    match parse_api_kwargs(vars(_parser.parse_args(args))):
        case Ok(result):
            return result
        case Err(e):
            print(f"Error parsing arguments: {e}")
            _parser.print_help()
            raise SystemExit(1)
