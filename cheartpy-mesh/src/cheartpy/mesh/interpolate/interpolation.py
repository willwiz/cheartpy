__all__ = ["INTERP_MAP", "interp_var_l2q", "make_l2qmap"]
from collections.abc import Mapping
from typing import cast

import numpy as np
from arraystubs import Arr1, Arr2
from cheartpy.io.api import chread_d, chwrite_d_utf
from cheartpy.mesh.struct import CheartMesh
from pytools.logging.api import NULL_LOGGER
from pytools.logging.trait import ILogger

from .maps import L2QMAPDICT, L2QTYPEDICT

type INTERP_MAP[T: np.integer] = Mapping[int, Arr1[T]]


def make_l2qmap[F: np.floating, I: np.integer](
    lin_mesh: CheartMesh[F, I],
    quad_mesh: CheartMesh[F, I],
) -> INTERP_MAP[I]:
    lin_top = lin_mesh.top
    quad_top = quad_mesh.top
    l2qmap = L2QMAPDICT.get(lin_top.TYPE)
    if l2qmap is None:
        msg = f"No map found for {lin_top.TYPE}. Topology to be interpolated must be linear"
        raise ValueError(msg)
    quad_elem = L2QTYPEDICT.get(lin_top.TYPE)
    if quad_elem is None:
        msg = (
            f"No quad topology found for {quad_top.TYPE}.Topology to be interpolated must be linear"
        )
        raise ValueError(msg)
    if lin_top.n != quad_top.n:
        msg = (
            f"The number of elements in the linear mesh ({lin_top.n}) and quad mesh ({quad_top.n})"
            "does not match "
        )
        raise ValueError(msg)
    interp_map: INTERP_MAP[I] = {}
    for i in range(lin_top.n):
        for j, e in enumerate(l2qmap):
            if quad_top.v[i, j] not in interp_map:
                interp_map[cast("int", quad_top.v[i, j])] = lin_top.v[i, list(e)]
    return interp_map


def interp_var_l2q[T: np.floating, I: np.integer](l2qmap: INTERP_MAP[I], lin: Arr2[T]) -> Arr2[T]:
    quad_data = np.zeros((len(l2qmap), lin.shape[1]), dtype=lin.dtype)
    for k, v in l2qmap.items():
        quad_data[k] = lin[v].mean(axis=0)
    return quad_data


def interpolate_var_on_lin_topology[I: np.integer](
    l2qmap: INTERP_MAP[I],
    lin_var: str,
    quad_var: str,
    log: ILogger = NULL_LOGGER,
) -> None:
    lin = chread_d(lin_var)
    if len(lin) == len(l2qmap):
        log.debug(f"Variable {lin_var} already interpolated")
        return
    quad = interp_var_l2q(l2qmap, lin)
    chwrite_d_utf(quad_var, quad)
