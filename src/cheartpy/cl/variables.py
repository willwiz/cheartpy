__all__ = [
    "create_lm_on_cl",
    "create_dm_on_cl",
    "Set_CLVAR_IC",
    "L2norm",
    "LL_interp",
]
import numpy as np
from typing import cast, overload
from ..var_types import *
from ..cheart.trait import IVariable
from ..cheart.api import create_variable
from .data import CLTopology, CLPartition


@overload
def create_lm_on_cl(
    cl: None, dim: int, ex_freq: int, set_bc: bool, sfx: str = "LM"
) -> None: ...
@overload
def create_lm_on_cl(
    cl: CLTopology, dim: int, ex_freq: int, set_bc: bool, sfx: str = "LM"
) -> IVariable: ...
def create_lm_on_cl(
    cl: CLTopology | None, dim: int, ex_freq: int, set_bc: bool, sfx: str = "LM"
) -> IVariable | None:
    if cl is None:
        return None
    lm = create_variable(f"{cl}{sfx}", cl.top_lm, dim, freq=ex_freq)
    if set_bc:
        print("Cannot set boundary conditions for a single variable")
    return lm


@overload
def create_dm_on_cl(
    cl: None, dim: int, ex_freq: int, set_bc: bool, sfx: str = "DM"
) -> None: ...
@overload
def create_dm_on_cl(
    cl: CLTopology, dim: int, ex_freq: int, set_bc: bool, sfx: str = "DM"
) -> IVariable: ...
def create_dm_on_cl(
    cl: CLTopology | None, dim: int, ex_freq: int, set_bc: bool, sfx: str = "DM"
) -> IVariable | None:
    if cl is None:
        return None
    lm = create_variable(f"{cl}{sfx}", None, dim, freq=ex_freq)
    if set_bc:
        print("Cannot set boundary conditions for a single variable")
    return lm


def Set_CLVAR_IC(var: IVariable | None, file: str) -> None:
    if var is None:
        return
    var.add_data(file)


def L2norm(x: Vec[f64]) -> float:
    return cast(float, x @ x)


def LL_basis(
    var: Mat[f64] | Vec[f64], nodes: Vec[f64], x: Vec[f64]
) -> tuple[Vec[int_t], Mat[f64]]:
    basis = {i: np.zeros_like(x) for i in range(2)}
    domain = (nodes[0] <= x) & (x <= nodes[1])
    basis[0][domain] = 1 - (x[domain] - nodes[0]) / (nodes[1] - nodes[0])
    basis[1][domain] = (x[domain] - nodes[0]) / (nodes[1] - nodes[0])
    return (domain, var[0] * basis[0][domain, None] + var[1] * basis[1][domain, None])


def LL_interp(top: CLPartition, var: Mat[f64], cl: Vec[f64]) -> Mat[f64]:
    x_bar = [LL_basis(var[elem], top.node[elem], cl) for elem in top.elem]
    res = np.zeros((len(cl), var.shape[1]), dtype=float)
    for k, v in x_bar:
        res[k] = v
    return res
