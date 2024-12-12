__all__ = [
    "CLBasis",
    "CLTopology",
    "CLTopologies",
    "CL_NODAL_LM_TYPE",
    "CLPartition",
    "CLNodalData",
    "PatchNode2ElemMap",
]
import dataclasses as dc
from typing import Mapping, TypedDict
from ..cheart.trait import IExpression, IVariable, ICheartTopology
from ..cheart_mesh.data import CheartMesh
from ..var_types import *

CL_NODAL_LM_TYPE = Mapping[int, IVariable]


@dc.dataclass(slots=True)
class CLPartition:
    prefix: str
    in_surf: int
    nn: int
    ne: int
    n_prefix: Mapping[int, str]
    e_prefix: Mapping[int, str]
    node: Vec[f64]
    elem: Mat[int_t]
    support: Mat[f64]

    def __repr__(self) -> str:
        return self.prefix


class CLNodalData(TypedDict):
    file: str
    mesh: CheartMesh
    n: Mat[f64]


@dc.dataclass(slots=True)
class PatchNode2ElemMap:
    i: Vec[int_t]  # global index of nodes in surface
    x: Vec[f64]  # cl value of nodes in surface
    n2e_map: Mapping[int, list[int]]


@dc.dataclass(slots=True)
class CLBasis:
    k: str
    t: ICheartTopology
    x: IVariable
    p: IExpression
    m: IExpression


@dc.dataclass(slots=True)
class CLTopology:
    prefix: str
    in_surf: int
    nn: int
    ne: int
    top_i: ICheartTopology
    top_lm: ICheartTopology
    support: IVariable
    elem: IVariable
    basis: IExpression
    b_vec: IExpression

    def __repr__(self) -> str:
        return self.prefix


@dc.dataclass(slots=True)
class CLTopologies:
    prefix: str
    in_surf: int
    nn: int
    ne: int
    field: IVariable
    top: ICheartTopology
    N: Mapping[int, CLBasis]

    def __repr__(self) -> str:
        return self.prefix
