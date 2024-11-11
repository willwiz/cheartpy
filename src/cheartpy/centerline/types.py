import dataclasses as dc
from typing import Mapping, TypedDict
from ..cheart_core.interface import IExpression
from ..var_types import *


@dc.dataclass(slots=True)
class CenterLinePartition:
    n: int
    prefix: Mapping[int, str]
    basis: Mapping[int, tuple[float, float, float]]


@dc.dataclass(slots=True)
class CLTopology:
    nn: int
    ne: int
    node_prefix: Mapping[int, str]
    elem_prefix: Mapping[int, str]
    node: Vec[f64]
    elem: Mat[i32]
    support: Mat[f64]


@dc.dataclass(slots=True)
class PatchNode2ElemMap:
    i: Vec[i32]  # global index of nodes in surface
    x: Vec[f64]  # cl value of nodes in surface
    n2e_map: Mapping[int, list[int]]


class CLBasisExpressions(TypedDict):
    pelem: Mapping[int, IExpression]
    melem: Mapping[int, IExpression]
