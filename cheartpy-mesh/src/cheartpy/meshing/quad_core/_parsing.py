import argparse
from typing import TYPE_CHECKING, Any, TypedDict
from warnings import warn

from pydantic import BaseModel, ValidationError
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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


class _ArgsModel(BaseModel):
    prefix: str
    xn: int
    yn: int


class _KwargsModel(BaseModel):
    shape: tuple[float, float]
    offset: tuple[float, float]


class SquareArgs(TypedDict, total=True):
    prefix: str
    xn: int
    yn: int


class SquareKwargs(TypedDict, total=False):
    shape: tuple[float, float]
    offset: tuple[float, float]


def parse_square_args(args: Mapping[str, Any]) -> Result[tuple[SquareArgs, SquareKwargs]]:
    try:
        square_args = _ArgsModel(**args)
    except ValidationError as e:
        return Err(e)
    try:
        square_kwargs = _KwargsModel(**args)
    except ValidationError as e:
        return Err(e)
    return Ok((SquareArgs(**square_args.model_dump()), SquareKwargs(**square_kwargs.model_dump())))


def get_square_args(args: Sequence[str] | None = None) -> tuple[SquareArgs, SquareKwargs]:
    namespace = square_parser.parse_args(args)
    match parse_square_args(vars(namespace)):
        case Ok(result):
            return result
        case Err(e):
            warn(f"Error parsing arguments: {e}.", stacklevel=2)
            square_parser.print_help()
            raise SystemExit(1)
