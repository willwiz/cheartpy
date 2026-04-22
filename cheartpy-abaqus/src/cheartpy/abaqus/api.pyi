# ruff: noqa: PYI011
from collections.abc import Sequence
from pathlib import Path
from typing import Unpack

import numpy as np
from cheartpy.mesh import CheartMesh
from pytools.arrays import DType
from pytools.result import Result

from .parsing import AbaqusAPIKwargs

def create_cheartmesh_from_abaqus_api[F: np.floating, I: np.integer](
    files: Sequence[Path | str],
    *,
    ftype: DType[F] = np.float64,
    dtype: DType[I] = np.intp,
    **kwargs: Unpack[AbaqusAPIKwargs],
) -> Result[CheartMesh[F, I]]: ...
