import argparse
from argparse import RawTextHelpFormatter
from typing import TYPE_CHECKING

from ._types import ParsedInput

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


def parse_cmdline_args(args: Sequence[str] | None = None) -> ParsedInput:
    return _parser.parse_args(
        args,
        namespace=ParsedInput(
            [], prefix=None, dim=3, topology=[], boundary=None, add_mask=[], cores=1
        ),
    )
