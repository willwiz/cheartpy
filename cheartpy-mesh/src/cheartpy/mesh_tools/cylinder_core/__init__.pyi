import argparse
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Unpack, overload

import numpy as np
from pytools.arrays import T3, ToFloat, ToInt
from pytools.result import Result

from cheartpy.mesh import CheartMesh

from ._parsing import CylinderArgs, CylinderKwargs

cylinder_parser: argparse.ArgumentParser

def get_cylinder_args(args: Sequence[str] | None = None) -> tuple[CylinderArgs, CylinderKwargs]: ...
@overload
def create_cylinder_mesh(
    shape: tuple[ToFloat, ToFloat, ToFloat, ToFloat],
    dim: T3[ToInt],
    axis: Literal["x", "y", "z"],
) -> tuple[CheartMesh[np.float64, np.intc], None]: ...
@overload
def create_cylinder_mesh(
    shape: tuple[ToFloat, ToFloat, ToFloat, ToFloat],
    dim: T3[ToInt],
    axis: Literal["x", "y", "z"],
    make_quad: Literal[False],
) -> tuple[CheartMesh[np.float64, np.intc], None]: ...
@overload
def create_cylinder_mesh(
    shape: tuple[ToFloat, ToFloat, ToFloat, ToFloat],
    dim: T3[ToInt],
    axis: Literal["x", "y", "z"],
    make_quad: Literal[True],
) -> tuple[CheartMesh[np.float64, np.intc], CheartMesh[np.float64, np.intc]]: ...
def make_cylinder_cli(
    args: CylinderArgs, **kwargs: Unpack[CylinderKwargs]
) -> tuple[CheartMesh[np.float64, np.intc], CheartMesh[np.float64, np.intc] | None]: ...
def parse_cylinder_args(args: Mapping[str, Any]) -> Result[tuple[CylinderArgs, CylinderKwargs]]: ...
