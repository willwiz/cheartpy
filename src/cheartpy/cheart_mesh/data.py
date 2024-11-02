__all__ = [
    "_CheartMeshSpace",
    "_CheartMeshTopology",
    "_CheartMeshPatch",
    "_CheartMeshBoundary",
    "CheartMesh",
    "create_bnd_surf",
]
import dataclasses as dc
import numpy as np
from ..var_types import Mat, Vec, f64, i32
from .elements import VtkType
from .io import *


@dc.dataclass(slots=True)
class _CheartMeshSpace:
    n: int
    v: Mat[f64]

    def save(self, name: str) -> None:
        CHWrite_d_utf(name, self.v)


@dc.dataclass(slots=True)
class _CheartMeshTopology:
    n: int
    v: Mat[i32]
    TYPE: VtkType

    def save(self, name: str) -> None:
        CHWrite_t_utf(name, self.v + 1, self.n, self.v.max() + 1)


@dc.dataclass(slots=True)
class _CheartMeshPatch:
    tag: int
    n: int
    k: Vec[i32]
    v: Mat[i32]

    def to_array(self) -> Mat[i32]:
        res = np.pad(self.v + 1, ((0, 0), (1, 1)))
        res[:, 0] = self.k + 1
        res[:, -1] = self.tag
        return res


@dc.dataclass(slots=True)
class _CheartMeshBoundary:
    n: int
    v: dict[str | int, _CheartMeshPatch]
    TYPE: VtkType

    def save(self, name: str) -> None:
        data = np.concatenate([v.to_array() for v in self.v.values()], axis=0)
        CHWrite_iarr_utf(name, data)


@dc.dataclass(slots=True)
class CheartMesh:
    space: _CheartMeshSpace
    top: _CheartMeshTopology
    bnd: _CheartMeshBoundary | None

    def save(self, prefix: str, forced: bool = False) -> None:
        if check_for_meshes("prefix") and not forced:
            return
        self.space.save(f"{prefix}_FE.X")
        self.top.save(f"{prefix}_FE.T")
        if self.bnd is not None:
            self.bnd.save(f"{prefix}_FE.B")


def create_bnd_surf(v: Mat[i32], tag: int) -> _CheartMeshPatch:
    bnd = v[v[:, -1] == tag, :-1]
    elems = bnd[:, 0] - 1
    nodes = bnd[:, 1:] - 1
    return _CheartMeshPatch(tag, len(bnd), elems, nodes)
