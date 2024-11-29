__all__ = [
    "LL_interp",
    "L2norm",
    "create_clbasis_exprs",
    "create_lms_on_cl",
]
import numpy as np
from typing import Mapping, TypedDict, cast
from collections import defaultdict
from .types import *
from ..cheart.trait import IVariable, IExpression
from ..cheart.impl import Expression
from ..cheart.api import create_variable
from ..cheart_mesh.data import *
from ..var_types import *
from ..tools.basiclogging import *


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


def LL_expr(
    name: str, v: IVariable, b: tuple[float, float, float] | Vec[f64]
) -> IExpression:
    basis = Expression(
        name,
        [
            f"max(min(({v}{-b[0]:+.8g})/({b[1] - b[0]:.8g}),({b[2]:.8g}-{v})/({b[2] - b[1]:.8g})), 0)"
        ],
    )
    basis.add_deps(v)
    return basis


def create_clbasis_exprs(
    var: IVariable,
    cl: CLPartition,
) -> CLBasisExpressions:
    pelem = {i: LL_expr(f"{s}B_p", var, cl.support[i]) for i, s in cl.n_prefix.items()}
    melem = {i: Expression(f"{s}B_m", [f"-{pelem[i]}"]) for i, s in cl.n_prefix.items()}
    [m.add_deps(pelem[k]) for k, m in melem.items()]
    return {"p": pelem, "m": melem}


def LL_str(v: IVariable, b: tuple[float, float, float] | Vec[f64]) -> str:
    return f"max(min(({v}{-b[0]:+.8g})/({b[1] - b[0]:.8g}),({b[2]:.8g}-{v})/({b[2] - b[1]:.8g})), 0)"


def create_line_top_expr(
    prefix: str,
    var: IVariable,
    cl: CLPartition,
) -> IExpression:
    return Expression(
        prefix,
        [LL_str(var, b) for b in cl.support],
    )


# def create_clbasis_expr(
#     prefix: str,
#     field: IVariable,
#     cl: CLPartition,
# ) -> IExpression:
#     return Expression(
#         prefix,
#         [
#             f"max(min(({field}{-b[0]:+.8g})/({b[1] - b[0]:.8g}),({b[2]:.8g}-{field})/({b[2] - b[1]:.8g})), 0)"
#         ],
#     )


def create_boundarynode_map(cl: Mat[f64], b: CheartMeshPatch) -> PatchNode2ElemMap:
    n2p_map: Mapping[int, list[int]] = defaultdict(list)
    for k, vs in enumerate(b.v):
        for v in vs:
            n2p_map[v].append(k)
    space_key = np.fromiter(n2p_map.keys(), dtype=int)
    return PatchNode2ElemMap(space_key, cl[space_key, 0], n2p_map)


def get_boundaryelems_in_clrange(
    map: PatchNode2ElemMap,
    domain: tuple[float, float] | Vec[f64],
) -> Vec[int_t]:
    nodes = map.i[((map.x - domain[0]) > 1.0e-8) & (domain[1] - map.x > 1.0e-8)]
    elems = np.fromiter(set([v for i in nodes for v in map.n2e_map[i]]), dtype=int)
    return elems


class CLNodalData(TypedDict):
    file: str
    mesh: CheartMesh
    n: Mat[f64]


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
