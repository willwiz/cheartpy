import argparse
from typing import Literal, TypedDict

cylinder_parser = argparse.ArgumentParser("cylinder", description="Make a cylinder")
cylinder_parser.add_argument(
    "--prefix",
    "-p",
    type=str,
    default="cube",
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


class CylinderArgs(TypedDict, total=True):
    rn: int
    qn: int
    zn: int
    rin: float
    rout: float
    length: float
    base: float


class CylinderKwargs(TypedDict, total=False):
    prefix: str
    axis: Literal["x", "y", "z"]
    make_quad: bool


def get_cylinder_args(args: list[str] | None = None) -> tuple[CylinderArgs, CylinderKwargs]:
    namespace = cylinder_parser.parse_args(args)
    _args_dict: CylinderArgs = {
        "rn": namespace.rn,
        "qn": namespace.qn,
        "zn": namespace.zn,
        "rin": namespace.rin,
        "rout": namespace.rout,
        "length": namespace.length,
        "base": namespace.base,
    }
    _kwargs_dict: CylinderKwargs = {
        "prefix": namespace.prefix,
        "axis": namespace.axis,
        "make_quad": namespace.make_quad,
    }
    return _args_dict, _kwargs_dict
