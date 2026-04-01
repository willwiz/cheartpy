import argparse
from collections.abc import Sequence
from typing import Unpack

import numpy as np
from pytools.arrays import T3, ToFloat, ToInt

from cheartpy.mesh import CheartMesh

from ._parsing import BlockArgs, BlockKwargs

block_parser: argparse.ArgumentParser

def get_block_args(args: Sequence[str] | None = None) -> tuple[BlockArgs, BlockKwargs]: ...
def create_hex_mesh(
    dim: T3[ToInt], shape: T3[ToFloat] = (1.0, 1.0, 1.0), shift: T3[float] = (0.0, 0.0, 0.0)
) -> CheartMesh[np.float64, np.intc]: ...
def make_block_api(
    args: BlockArgs, **kwargs: Unpack[BlockKwargs]
) -> CheartMesh[np.float64, np.intc]: ...
