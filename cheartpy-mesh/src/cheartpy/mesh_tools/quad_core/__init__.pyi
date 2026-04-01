import argparse
from collections.abc import Sequence
from typing import Unpack

import numpy as np
from pytools.arrays import T2, ToFloat, ToInt

from cheartpy.mesh import CheartMesh

from ._parsing import SquareArgs, SquareKwargs

square_parser: argparse.ArgumentParser

def get_square_args(args: Sequence[str] | None = None) -> tuple[SquareArgs, SquareKwargs]: ...
def make_square_api(
    dim: SquareArgs, **kwargs: Unpack[SquareKwargs]
) -> CheartMesh[np.float64, np.intc]: ...
def create_square_mesh(
    dim: T2[ToInt], shape: T2[ToFloat] = (1.0, 1.0), shift: T2[ToFloat] = (0.0, 0.0)
) -> CheartMesh[np.float64, np.intc]: ...
