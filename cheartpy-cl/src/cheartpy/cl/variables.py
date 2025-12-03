from typing import TYPE_CHECKING, overload

import numpy as np
from cheartpy.fe.api import create_variable

if TYPE_CHECKING:
    from cheartpy.fe.trait import IVariable
    from pytools.arrays import A1, A2

    from .struct import CLPartition, CLTopology

__all__ = [
    "create_dm_on_cl",
    "create_lm_on_cl",
    "l2_norm",
    "ll_interp",
    "set_clvar_ic",
]


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


def l2_norm(x: A1[np.floating]) -> float:
    return float(x @ x)


def ll_basis[F: np.floating](
    var: A2[F] | A1[F],
    nodes: A1[F],
    x: A1[F],
) -> tuple[A1[np.intc], A2[F]]:
    basis = {i: np.zeros_like(x) for i in range(2)}
    domain = (nodes[0] <= x) & (x <= nodes[1]).astype(np.intc)
    basis[0][domain] = 1 - (x[domain] - nodes[0]) / (nodes[1] - nodes[0])
    basis[1][domain] = (x[domain] - nodes[0]) / (nodes[1] - nodes[0])
    return (domain, var[0] * basis[0][domain, None] + var[1] * basis[1][domain, None])


def ll_interp[F: np.floating](
    top: CLPartition[F, np.integer],
    var: A2[F],
    cl: A1[F],
) -> A2[F]:
    x_bar = [ll_basis(var[elem], top.node[elem], cl) for elem in top.elem]
    res = np.zeros((len(cl), var.shape[1]), dtype=float)
    for k, v in x_bar:
        res[k] = v
    return res
