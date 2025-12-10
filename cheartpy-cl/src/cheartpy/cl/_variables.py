from typing import TYPE_CHECKING, TypedDict, Unpack, overload

import numpy as np
from cheartpy.fe.api import create_variable

if TYPE_CHECKING:
    from pathlib import Path

    from cheartpy.fe.trait import IVariable
    from pytools.arrays import A1, A2

    from .struct import CLPartition, CLStructure


class _VaribleKwaargs(TypedDict, total=False):
    freq: int
    data: Path | str


@overload
def create_lm_on_cl(
    cl: None, dim: int, sfx: str = "LM", **kwargs: Unpack[_VaribleKwaargs]
) -> None: ...
@overload
def create_lm_on_cl(
    cl: CLStructure, dim: int, sfx: str = "LM", **kwargs: Unpack[_VaribleKwaargs]
) -> IVariable: ...
def create_lm_on_cl(
    cl: CLStructure | None, dim: int, sfx: str = "LM", **kwargs: Unpack[_VaribleKwaargs]
) -> IVariable | None:
    if cl is None:
        return None
    return create_variable(
        f"{cl}{sfx}", cl.top_lm, dim, freq=kwargs.get("freq", 1), data=kwargs.get("data")
    )


@overload
def create_dm_on_cl(
    cl: None, dim: int, sfx: str = "DM", **kwargs: Unpack[_VaribleKwaargs]
) -> None: ...
@overload
def create_dm_on_cl(
    cl: CLStructure, dim: int, sfx: str = "DM", **kwargs: Unpack[_VaribleKwaargs]
) -> IVariable: ...
def create_dm_on_cl(
    cl: CLStructure | None, dim: int, sfx: str = "DM", **kwargs: Unpack[_VaribleKwaargs]
) -> IVariable | None:
    if cl is None:
        return None
    return create_variable(
        f"{cl}{sfx}", None, dim, freq=kwargs.get("freq", 1), data=kwargs.get("data")
    )


def set_clvar_ic(var: IVariable | None, file: Path | str) -> None:
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
