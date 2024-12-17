__all__ = [
    "create_lm_on_cl",
    "create_dm_on_cl",
    "Set_CLVAR_IC",
    "L2norm",
    "LL_interp",
]
import numpy as np
from typing import cast, Mapping, overload
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
) -> Mapping[int, Mat[f64]]:
    basis_func = {i: np.zeros_like(x) for i in range(2)}
    in_domain = (nodes[0] <= x) & (x <= nodes[1])
    basis_func[0][in_domain] = 1 - (x[in_domain] - nodes[0]) / (nodes[1] - nodes[0])
    basis_func[1][in_domain] = (x[in_domain] - nodes[0]) / (nodes[1] - nodes[0])
    return {k: var[k] * v[:, None] for k, v in basis_func.items()}


def LL_interp(top: CLPartition, var: Mat[f64] | Vec[f64], cl: Vec[f64]) -> Mat[f64]:
    x_bar = [
        v for elem in top.elem for v in LL_basis(var[elem], top.node[elem], cl).values()
    ]
    return cast(Mat[f64], sum(x_bar))
