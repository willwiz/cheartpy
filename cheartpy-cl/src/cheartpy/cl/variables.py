__all__ = [
    "create_dm_on_cl",
    "create_lm_on_cl",
    "l2_norm",
    "ll_interp",
    "set_clvar_ic",
]
from typing import overload

import numpy as np
from arraystubs import Arr1, Arr2

from cheartpy.cheart.api import create_variable
from cheartpy.cheart.trait import IVariable

from .struct import CLPartition, CLTopology


@overload
def create_lm_on_cl(cl: None, dim: int, ex_freq: int, sfx: str = "LM") -> None: ...
@overload
def create_lm_on_cl(cl: CLTopology, dim: int, ex_freq: int, sfx: str = "LM") -> IVariable: ...
def create_lm_on_cl(
    cl: CLTopology | None,
    dim: int,
    ex_freq: int,
    sfx: str = "LM",
) -> IVariable | None:
    if cl is None:
        return None
    return create_variable(f"{cl}{sfx}", cl.top_lm, dim, freq=ex_freq)


@overload
def create_dm_on_cl(cl: None, dim: int, ex_freq: int, sfx: str = "DM") -> None: ...
@overload
def create_dm_on_cl(cl: CLTopology, dim: int, ex_freq: int, sfx: str = "DM") -> IVariable: ...
def create_dm_on_cl(
    cl: CLTopology | None,
    dim: int,
    ex_freq: int,
    sfx: str = "DM",
) -> IVariable | None:
    if cl is None:
        return None
    return create_variable(f"{cl}{sfx}", None, dim, freq=ex_freq)


def set_clvar_ic(var: IVariable | None, file: str) -> None:
    """Set initial condition for a CL variable from a file."""
    if var is None:
        return
    var.add_data(file)


def l2_norm(x: Arr1[np.floating]) -> float:
    return float(x @ x)


def ll_basis[F: np.floating](
    var: Arr2[F] | Arr1[F],
    nodes: Arr1[F],
    x: Arr1[F],
) -> tuple[Arr1[np.intc], Arr2[F]]:
    basis = {i: np.zeros_like(x) for i in range(2)}
    domain = (nodes[0] <= x) & (x <= nodes[1]).astype(np.intc)
    basis[0][domain] = 1 - (x[domain] - nodes[0]) / (nodes[1] - nodes[0])
    basis[1][domain] = (x[domain] - nodes[0]) / (nodes[1] - nodes[0])
    return (domain, var[0] * basis[0][domain, None] + var[1] * basis[1][domain, None])


def ll_interp[F: np.floating](
    top: CLPartition[F, np.integer],
    var: Arr2[F],
    cl: Arr1[F],
) -> Arr2[F]:
    x_bar = [ll_basis(var[elem], top.node[elem], cl) for elem in top.elem]
    res = np.zeros((len(cl), var.shape[1]), dtype=float)
    for k, v in x_bar:
        res[k] = v
    return res
