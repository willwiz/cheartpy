import argparse
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import ToFloat, ToInt

square_parser = argparse.ArgumentParser("square", description="Make a square")
square_parser.add_argument(
    "--prefix",
    "-p",
    type=str,
    default="square",
    help="Prefix for mesh.",
)
square_parser.add_argument(
    "--shape",
    "-s",
    type=float,
    nargs=2,
    default=[1, 1],
    metavar=("Lx", "Ly"),
    help="lengths",
)
square_parser.add_argument(
    "--offset",
    type=float,
    nargs=2,
    default=[0, 0],
    metavar=("xoff", "yoff"),
    help="starting corner should be shifted by",
)
square_parser.add_argument("xn", type=int, help="number of elements in x")
square_parser.add_argument("yn", type=int, help="number of elements in y")


class SquareKwargs(TypedDict, total=False):
    prefix: str
    shape: tuple[ToFloat, ToFloat]
    offset: tuple[ToFloat, ToFloat]


class SquareArgs(TypedDict, total=True):
    xn: ToInt
    yn: ToInt


def get_square_args(args: Sequence[str] | None = None) -> tuple[SquareArgs, SquareKwargs]:
    namespace = square_parser.parse_args(args)
    _args_dict: SquareArgs = {
        "xn": namespace.xn,
        "yn": namespace.yn,
    }
    _kwargs_dict: SquareKwargs = {
        "prefix": namespace.prefix,
        "shape": tuple(namespace.size),
        "offset": tuple(namespace.offset),
    }
    return _args_dict, _kwargs_dict
