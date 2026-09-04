import argparse
from typing import TYPE_CHECKING, Any, Literal, Required, TypedDict
from warnings import warn

from pydantic import BaseModel, ValidationError
from pytools.result import Err, Ok, Result

if TYPE_CHECKING:
    from collections.abc import Mapping


cylinder_parser = argparse.ArgumentParser("cylinder", description="Make a cylinder")
cylinder_parser.add_argument(
    "--prefix",
    "-p",
    type=str,
    default="cylinder",
    help="Prefix for saved file.",
)
cylinder_parser.add_argument("-l", "--length", type=float, default=1, help="long axis length")
cylinder_parser.add_argument("-b", "--base", type=float, default=0, help="starting location")
cylinder_parser.add_argument(
    "--axis",
    "-a",
    type=str,
    default="z",
    choices={"x", "y", "z"},
    help="Which cartesian axis should the central axis be in.",
)
cylinder_parser.add_argument("--make-quad", action="store_true", help="auto make a quad mesh")
cylinder_parser.add_argument("rin", type=float, help="number of elements in r")
cylinder_parser.add_argument("rout", type=float, help="number of elements in r")
cylinder_parser.add_argument("rn", type=int, help="number of elements in r")
cylinder_parser.add_argument("qn", type=int, help="number of elements in theta")
cylinder_parser.add_argument("zn", type=int, help="number of elements in z")


class _ArgsModel(BaseModel):
    prefix: str
    rin: float
    rout: float
    length: float
    base: float
    rn: int
    qn: int
    zn: int


class _KwargsModel(BaseModel):
    axis: Literal["x", "y", "z"]
    make_quad: bool


class CylinderArgs(TypedDict, total=True):
    prefix: str
    rn: int
    qn: int
    zn: int
    rin: float
    rout: float
    length: float
    base: float


class CylinderKwargs(TypedDict, total=False):
    axis: Required[Literal["x", "y", "z"]]
    make_quad: bool


def parse_cylinder_args(args: Mapping[str, Any]) -> Result[tuple[CylinderArgs, CylinderKwargs]]:
    try:
        cylinder_args = _ArgsModel(**args)
    except ValidationError as e:
        return Err(e)
    try:
        cylinder_kwargs = _KwargsModel(**args)
    except ValidationError as e:
        return Err(e)
    return Ok(
        (CylinderArgs(**cylinder_args.model_dump()), CylinderKwargs(**cylinder_kwargs.model_dump()))
    )


def get_cylinder_args(args: list[str] | None = None) -> tuple[CylinderArgs, CylinderKwargs]:
    namespace = cylinder_parser.parse_args(args)
    match parse_cylinder_args(vars(namespace)):
        case Ok(result):
            return result
        case Err(e):
            warn(f"Error parsing cylinder args: {e}", stacklevel=2)
            cylinder_parser.print_help()
            raise SystemExit(1)
