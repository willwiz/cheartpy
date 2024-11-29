__all__ = [
    "CheartMeshSpace",
    "CheartMeshTopology",
    "CheartMeshPatch",
    "CheartMeshBoundary",
    "CheartMesh",
    "create_bnd_surf",
]
import dataclasses as dc
from typing import Mapping
import numpy as np
from ..var_types import Mat, Vec, f64, int_t
from .elements import VtkType
from .io import *


@dc.dataclass(slots=True)
class CheartMeshSpace:
    n: int
    v: Mat[f64]

    def save(self, name: str) -> None:
        CHWrite_d_utf(name, self.v)


@dc.dataclass(slots=True)
class CheartMeshTopology:
    n: int
    v: Mat[int_t]
    TYPE: VtkType

    def save(self, name: str) -> None:
        CHWrite_t_utf(name, self.v + 1, self.v.max() + 1)


@dc.dataclass(slots=True)
class CheartMeshPatch:
    tag: int
    n: int
    k: Vec[int_t]
    v: Mat[int_t]

    def to_array(self) -> Mat[int_t]:
        res = np.pad(self.v + 1, ((0, 0), (1, 1)))
        res[:, 0] = self.k + 1
        res[:, -1] = self.tag
        return res


@dc.dataclass(slots=True)
class CheartMeshBoundary:
    n: int
    v: Mapping[str | int, CheartMeshPatch]
    TYPE: VtkType

    def save(self, name: str) -> None:
        data = np.concatenate([v.to_array() for v in self.v.values()], axis=0)
        CHWrite_iarr_utf(name, data)


@dc.dataclass(slots=True)
class CheartMesh:
    space: CheartMeshSpace
    top: CheartMeshTopology
    bnd: CheartMeshBoundary | None

    def save(self, prefix: str, forced: bool = False) -> None:
        if check_for_meshes("prefix") and not forced:
            return
        prefix = fix_suffix(prefix)
        self.space.save(f"{prefix}X")
        self.top.save(f"{prefix}T")
        if self.bnd is not None:
            self.bnd.save(f"{prefix}B")


def create_bnd_surf(v: Mat[int_t], tag: int) -> CheartMeshPatch:
    bnd = v[v[:, -1] == tag, :-1]
    elems = bnd[:, 0] - 1
    nodes = bnd[:, 1:] - 1
    return CheartMeshPatch(tag, len(bnd), elems, nodes)
