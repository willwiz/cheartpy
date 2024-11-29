__all__ = ["create_lms_on_cl", "L2norm", "LL_interp"]
import numpy as np
from typing import cast, Mapping
from ..var_types import *
from ..cheart.trait import IVariable
from ..cheart.api import create_variable
from .data import CLTopology, CLPartition


def create_lms_on_cl(
    prefix: str, cl: CLTopology | None, dim: int, ex_freq: int, set_bc: bool
) -> Mapping[int, IVariable]:
    if cl is None:
        return {}
    lms = {
        k: create_variable(f"{v.k}{prefix}", None, dim, freq=ex_freq)
        for k, v in cl.N.items()
    }
    if set_bc:
        keys = sorted(lms.keys())
        lms[keys[0]] = lms[keys[1]]
        lms[keys[-1]] = lms[keys[-2]]
    return lms


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
