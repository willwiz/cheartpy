from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pytools.arrays import DType

    from ._types import CLDef


def get_cl_ftype[F: np.floating = np.float64](defn: CLDef[F]) -> DType[F]:
    match defn:
        case {"nodes": nodes}:
            return nodes.dtype
        case {"a_z": (_, dtype)}:
            return dtype
        case _:
            msg = "Unreachable"
            raise RuntimeError(msg)
