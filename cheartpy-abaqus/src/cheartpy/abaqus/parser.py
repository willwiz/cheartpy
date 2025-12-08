import argparse
from argparse import RawTextHelpFormatter
from pathlib import Path
from typing import TYPE_CHECKING

from .struct import InputArgs, Mask

if TYPE_CHECKING:
    from collections.abc import Sequence

################################################################################################
# Check if multiprocessing is available


parser = argparse.ArgumentParser(
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
parser.add_argument(
    "input",
    nargs="+",
    type=str,
    help="""Name of the .inp file containing the Abaqus mesh. If given after the
    optional arguments -t or -b, -- should be inserted in between to delineate.
    """,
)
parser.add_argument(
    "-d",
    "--dim",
    type=int,
    default=3,
    help="""dimension of the mesh, default 3""",
)
parser.add_argument(
    "-p",
    "--prefix",
    type=str,
    default=None,
    help="""Give the prefix for the output files.""",
)
parser.add_argument(
    "-t",
    "--topology",
    nargs="+",
    default=None,
    help="""Define which volume will be used as the topology. If multiple are given,
    they are appended. E.g.,
    --topology Volume1
    --topology Volume1 Volume2 Volume3 ...
    """,
)
parser.add_argument(
    "-b",
    "--boundary",
    action="append",
    nargs="+",
    default=None,
    help="""Set a boundary give the name of the element and label or name, appended
    numerals, and label. E.g.,
    --boundary Surf1  label
    --boundary Surf1 Surf2 ... label
    """,
)
parser.add_argument(
    "--add-mask",
    action="append",
    nargs="+",
    default=None,
    help="""Add a mask with an given element""",
)
parser.add_argument(
    "-c",
    "--cores",
    default=1,
    type=int,
    help="""Enable multiprocessing with n cores
    """,
)

_MASK_ARG_LEN = 3


def gather_masks(
    cmd_arg_masks: Sequence[Sequence[str]] | None,
) -> dict[str, Mask] | None:
    if cmd_arg_masks is None:
        return None
    masks: dict[str, Mask] = {}
    for m in cmd_arg_masks:
        if len(m) < _MASK_ARG_LEN:
            msg = (
                ">>>ERROR: A Mask must have at least 3 args, e.g., name1, name2, ..., value, output"
            )
            raise ValueError(msg)
        masks[m[-1]] = Mask(m[-1], m[-2], m[:-2])
    return masks


def split_argslist_to_nameddict(
    varlist: Sequence[Sequence[str]] | None,
) -> dict[int, Sequence[str]] | None:
    if varlist is None:
        return None
    var: dict[int, Sequence[str]] = {}
    for items in varlist:
        if not len(items) > 1:
            msg = ">>>ERROR: Boundary or Topology must have at least 2 items, elem and label."
            raise ValueError(msg)
        var[int(items[-1])] = items[:-1]
    return var


def check_args(args: argparse.Namespace) -> InputArgs:
    if args.prefix is None:
        name, _ = Path(args.input[0]).stem
    else:
        name: str = args.prefix
    boundary = split_argslist_to_nameddict(args.boundary)
    masks = gather_masks(args.add_mask)
    return InputArgs(
        args.input,
        name,
        args.dim,
        args.topology,
        boundary,
        masks,
        args.cores,
    )
