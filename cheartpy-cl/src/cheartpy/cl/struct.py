import dataclasses as dc
from collections.abc import Mapping
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from cheartpy.fe.trait import ICheartTopology, IExpression, IVariable

if TYPE_CHECKING:
    from pathlib import Path

    from cheartpy.mesh.struct import CheartMesh
    from pytools.arrays import A1, A2

__all__ = [
    "CL_NODAL_LM_TYPE",
    "CLBasis",
    "CLNodalData",
    "CLPartition",
    "CLTopologies",
    "CLTopology",
    "PatchNode2ElemMap",
]
CL_NODAL_LM_TYPE = Mapping[int, IVariable]


@dc.dataclass(slots=True)
class CLPartition[F: np.floating, I: np.integer]:
    prefix: str
    in_surf: int
    nn: int
    ne: int
    n_prefix: Mapping[int, str]
    e_prefix: Mapping[int, str]
    node: A1[F]
    elem: A2[I]
    support: A2[F]

    def __repr__(self) -> str:
        return self.prefix


class CLNodalData[F: np.floating, I: np.integer](TypedDict):
    file: Path
    mesh: CheartMesh[F, I]
    n: A2[F]


@dc.dataclass(slots=True)
class PatchNode2ElemMap:
    i: A1[np.integer]  # global index of nodes in surface
    x: A1[np.floating]  # cl value of nodes in surface
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
