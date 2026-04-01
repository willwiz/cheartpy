import argparse
from typing import Required, TypedDict

block_parser = argparse.ArgumentParser("block", description="Make a cube")
block_parser.add_argument(
    "--prefix",
    "-p",
    type=str,
    default="cube",
    help="Prefix for saved file.",
)
block_parser.add_argument(
    "--shape",
    "-s",
    type=float,
    nargs=3,
    default=[1, 1, 1],
    metavar=("Lx", "Ly", "Lz"),
    help="starting corner should be shifted by",
)
block_parser.add_argument(
    "--offset",
    "-o",
    type=float,
    nargs=3,
    default=[0, 0, 0],
    metavar=("x(0)", "y(0)", "z(0)"),
    help="starting corner should be shifted by",
)
block_parser.add_argument("xn", type=int, help="number of elements in x")
block_parser.add_argument("yn", type=int, help="number of elements in y")
block_parser.add_argument("zn", type=int, help="number of elements in z")


class BlockArgs(TypedDict, total=True):
    xn: int
    yn: int
    zn: int


class BlockKwargs(TypedDict, total=False):
    prefix: str
    shape: Required[tuple[float, float, float]]
    offset: Required[tuple[float, float, float]]


def get_block_args(args: list[str] | None = None) -> tuple[BlockArgs, BlockKwargs]:
    namespace = block_parser.parse_args(args)
    _args_dict: BlockArgs = {
        "xn": namespace.xn,
        "yn": namespace.yn,
        "zn": namespace.zn,
    }
    _kwargs_dict: BlockKwargs = {
        "prefix": namespace.prefix,
        "shape": tuple(namespace.shape),
        "offset": tuple(namespace.offset),
    }
    return _args_dict, _kwargs_dict
