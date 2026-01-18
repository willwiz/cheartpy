from pathlib import Path
from typing import TYPE_CHECKING, cast

import meshio
import numpy as np
from pytools.result import Err, Ok

if TYPE_CHECKING:
    from pytools.arrays import S2D, DType

    from ._data import MeshioMesh


def import_meshio[F: np.floating, I: np.integer](
    file: Path | str, _dtype: DType[I] = np.intp, ftype: DType[F] = np.float64
) -> Ok[MeshioMesh[S2D, F, I]] | Err:
    file = Path(file)
    try:
        mesh = cast("MeshioMesh[S2D, F, I]", meshio.read(file))  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    except meshio.ReadError as e:
        return Err(e)
    mesh.points = mesh.points.astype(ftype)
    return Ok(mesh)
