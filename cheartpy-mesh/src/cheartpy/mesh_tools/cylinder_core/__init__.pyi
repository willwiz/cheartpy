import argparse
from collections.abc import Sequence
from typing import Literal, Unpack, overload

import numpy as np
from pytools.arrays import T3

from cheartpy.mesh import CheartMesh

from ._parsing import CylinderArgs, CylinderKwargs

cylinder_parser: argparse.ArgumentParser

def get_cylinder_args(args: Sequence[str] | None = None) -> tuple[CylinderArgs, CylinderKwargs]: ...
@overload
def create_cylinder_mesh(
    shape: tuple[float, float, float, float],
    dim: T3[int],
    axis: Literal["x", "y", "z"],
) -> tuple[CheartMesh[np.float64, np.intc], None]: ...
@overload
def create_cylinder_mesh(
    shape: tuple[float, float, float, float],
    dim: T3[int],
    axis: Literal["x", "y", "z"],
    make_quad: Literal[False],
) -> tuple[CheartMesh[np.float64, np.intc], None]: ...
@overload
def create_cylinder_mesh(
    shape: tuple[float, float, float, float],
    dim: T3[int],
    axis: Literal["x", "y", "z"],
    make_quad: Literal[True],
) -> tuple[CheartMesh[np.float64, np.intc], CheartMesh[np.float64, np.intc]]: ...
def make_cylinder_api(
    args: CylinderArgs, **kwargs: Unpack[CylinderKwargs]
) -> tuple[CheartMesh[np.float64, np.intc], CheartMesh[np.float64, np.intc] | None]: ...
