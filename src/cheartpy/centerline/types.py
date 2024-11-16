__all__ = [
    "CLBasis",
    "CLTopology",
    "CL_LM_TYPE",
    "CLPartition",
    "PatchNode2ElemMap",
    "CLBasisExpressions",
]
import dataclasses as dc
from typing import Mapping, TypedDict
from ..cheart.trait import IExpression, IVariable, ICheartTopology
from ..var_types import *

CL_LM_TYPE = Mapping[int, IVariable]


@dc.dataclass(slots=True)
class CenterLinePartition:
    n: int
    prefix: Mapping[int, str]
    basis: Mapping[int, tuple[float, float, float]]


@dc.dataclass(slots=True)
class CLPartition:
    prefix: str
    in_surf: int
    nn: int
    ne: int
    n_prefix: Mapping[int, str]
    e_prefix: Mapping[int, str]
    node: Vec[f64]
    elem: Mat[i32]
    support: Mat[f64]


@dc.dataclass(slots=True)
class PatchNode2ElemMap:
    i: Vec[i32]  # global index of nodes in surface
    x: Vec[f64]  # cl value of nodes in surface
    n2e_map: Mapping[int, list[int]]


class CLBasisExpressions(TypedDict):
    p: Mapping[int, IExpression]
    m: Mapping[int, IExpression]


@dc.dataclass(slots=True)
class CLBasis:
    k: str
    t: ICheartTopology
    x: IVariable
    p: IExpression
    m: IExpression
    n_p: IExpression
    n_m: IExpression


@dc.dataclass(slots=True)
class CLTopology:
    prefix: str
    in_surf: int
    nn: int
    ne: int
    field: IVariable
    top: ICheartTopology
    N: Mapping[int, CLBasis]
