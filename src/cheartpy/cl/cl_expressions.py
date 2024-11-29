__all__ = ["create_clbasis_exprs"]
import numpy as np
from typing import Mapping
from collections import defaultdict
from ..cheart.trait import IVariable, IExpression
from ..cheart.impl import Expression
from ..cheart_mesh.data import *
from ..var_types import *
from .data import *


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
