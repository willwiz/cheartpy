__all__ = ["INTERP_MAP", "make_l2qmap", "interp_var_l2q"]
import numpy as np
from typing import Mapping, Sequence
from ...var_types import *
from ...cheart_mesh.data import CheartMesh
from .maps import L2QMAPDICT, L2QTYPEDICT

type INTERP_MAP = Mapping[int, Sequence[int]]


def make_l2qmap(lin_mesh: CheartMesh, quad_mesh: CheartMesh) -> INTERP_MAP:
    lin_top = lin_mesh.top
    quad_top = quad_mesh.top
    if not L2QTYPEDICT.get(lin_top.TYPE) is quad_top.TYPE:
        raise ValueError(
            f"Quad mesh ({quad_top.TYPE}) must be quadratic version of linear mesh ({lin_top.TYPE})"
        )
    if not lin_top.n == quad_top.n:
        raise ValueError(
            f"The number of elements in the linear mesh ({lin_top.n}) and quad mesh ({quad_top.n}) does not match "
        )
    L2Q = L2QMAPDICT[lin_top.TYPE]
    interp_map: INTERP_MAP = dict()
    for i in range(lin_top.n):
        for j, e in enumerate(L2Q):
            if quad_top.v[i, j] not in interp_map:
                interp_map[quad_top.v[i, j]] = [lin_top.v[i, k] for k in e]
    return interp_map


def interp_var_l2q(map: INTERP_MAP, lin: Mat[f64]):
    quad_data = np.zeros((len(map), lin.shape[1]), dtype=np.float64)
    for k, v in map.items():
        quad_data[k] = lin[v].mean(axis=0)
    return quad_data
