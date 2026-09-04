import argparse
from typing import TYPE_CHECKING, Any, Required, TypedDict
from warnings import warn

from pydantic import BaseModel, ValidationError
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pytools.arrays import T3

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


class _ArgsModel(BaseModel):
    prefix: str
    xn: int
    yn: int
    zn: int


class _KwargsModel(BaseModel):
    shape: tuple[float, float, float]
    offset: tuple[float, float, float]


class BlockArgs(TypedDict, total=True):
    prefix: str
    xn: int
    yn: int
    zn: int


class BlockKwargs(TypedDict, total=False):
    shape: Required[T3[float]]
    offset: Required[T3[float]]


def parse_block_args(args: Mapping[str, Any]) -> Result[tuple[BlockArgs, BlockKwargs]]:
    try:
        block_args = _ArgsModel(**args)
    except ValidationError as e:
        return Err(e)
    try:
        block_kwargs = _KwargsModel(**args)
    except ValidationError as e:
        return Err(e)
    return Ok((BlockArgs(**block_args.model_dump()), BlockKwargs(**block_kwargs.model_dump())))


def get_block_args(args: list[str] | None = None) -> tuple[BlockArgs, BlockKwargs]:
    namespace = block_parser.parse_args(args)
    match parse_block_args(vars(namespace)):
        case Ok(result):
            return result
        case Err(e):
            warn(f"Error parsing arguments: {e}.", stacklevel=2)
            block_parser.print_help()
            raise SystemExit(1)
