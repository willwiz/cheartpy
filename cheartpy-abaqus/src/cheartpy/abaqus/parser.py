import argparse
import dataclasses as dc
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import TYPE_CHECKING, overload

from pytools.result import Err, Ok

from ._struct import InputArgs, Mask

if TYPE_CHECKING:
    from collections.abc import Sequence

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
    "input",
    nargs="+",
    type=str,
    help="""Name of the .inp file containing the Abaqus mesh. If given after the
    optional arguments -t or -b, -- should be inserted in between to delineate.
    """,
)
_parser.add_argument(
    "-d",
    "--dim",
    type=int,
    help="""dimension of the mesh, default 3""",
)
_parser.add_argument(
    "-p",
    "--prefix",
    type=str,
    help="""Give the prefix for the output files.""",
)
_parser.add_argument(
    "-t",
    "--topology",
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
    "-b",
    "--boundary",
    type=str,
    action="append",
    nargs="+",
    default=None,
    help="""Set a boundary give the name of the element and label or name, appended
    numerals, and label. E.g.,
    --boundary Surf1  label
    --boundary Surf1 Surf2 ... label
    """,
)
_parser.add_argument(
    "--add-mask",
    type=str,
    action="append",
    nargs="+",
    default=None,
    help="""Add a mask with an given element""",
)
_parser.add_argument("-c", "--cores", type=int, help="""Enable multiprocessing with n cores""")

_MASK_ARG_LEN = 3


@dc.dataclass(slots=True)
class _AbaqusInput:
    inputs: Sequence[str]
    prefix: str | None
    dim: int
    topology: Sequence[str]
    boundary: Sequence[str] | None
    add_mask: Sequence[str]
    cores: int


def parse_cmdline_args(args: Sequence[str] | None = None) -> _AbaqusInput:
    return _parser.parse_args(
        args,
        namespace=_AbaqusInput(
            [], prefix=None, dim=3, topology=[], boundary=None, add_mask=[], cores=1
        ),
    )


@overload
def gather_masks(cmd_arg_masks: None) -> Ok[None]: ...
@overload
def gather_masks(cmd_arg_masks: Sequence[Sequence[str]]) -> Ok[dict[str, Mask]] | Err: ...
def gather_masks(
    cmd_arg_masks: Sequence[Sequence[str]] | None,
) -> Ok[dict[str, Mask]] | Ok[None] | Err:
    if cmd_arg_masks is None:
        return Ok(None)
    masks: dict[str, Mask] = {}
    for m in cmd_arg_masks:
        if len(m) < _MASK_ARG_LEN:
            msg = (
                ">>>ERROR: A Mask must have at least 3 args, e.g., name1, name2, ..., value, output"
            )
            return Err(ValueError(msg))
        masks[m[-1]] = Mask(m[-1], m[-2], m[:-2])
    return Ok(masks)


@overload
def split_argslist_to_nameddict(varlist: None) -> Ok[None]: ...
@overload
def split_argslist_to_nameddict(
    varlist: Sequence[Sequence[str]],
) -> Ok[dict[int, Sequence[str]]] | Err: ...
def split_argslist_to_nameddict(
    varlist: Sequence[Sequence[str]] | None,
) -> Ok[dict[int, Sequence[str]]] | Ok[None] | Err:
    if varlist is None:
        return Ok(None)
    var: dict[int, Sequence[str]] = {}
    for items in varlist:
        if not len(items) > 1:
            msg = ">>>ERROR: Boundary or Topology must have at least 2 items, elem and label."
            return Err(ValueError(msg))
        var[int(items[-1])] = items[:-1]
    return Ok(var)


def check_args(args: _AbaqusInput) -> Ok[InputArgs] | Err:
    name = Path(args.inputs[0]).stem if args.prefix is None else args.prefix
    match split_argslist_to_nameddict(args.boundary):
        case Ok(boundary):
            pass
        case Err(e):
            return Err(e)
    match gather_masks(args.add_mask):
        case Ok(masks):
            pass
        case Err(e):
            return Err(e)
    return Ok(InputArgs(args.inputs, name, args.dim, args.topology, boundary, masks, args.cores))
