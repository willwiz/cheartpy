import dataclasses as dc
import numpy as np
from ...var_types import Mat, Vec, f64, i32
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
    v: Mat[i32]
    TYPE: VtkType

    def save(self, name: str) -> None:
        CHWrite_t_utf(name, self.v + 1, self.n, self.v.max() + 1)


@dc.dataclass(slots=True)
class CheartMeshSurface:
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
class CheartMeshBoundary:
    n: int
    v: dict[str | int, CheartMeshSurface]
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
        self.space.save(f"{prefix}_FE.X")
        self.top.save(f"{prefix}_FE.T")
        if self.bnd is not None:
            self.bnd.save(f"{prefix}_FE.B")


def create_bnd_surf(v: Mat[i32], tag: int) -> CheartMeshSurface:
    bnd = v[v[:, -1] == tag, :-1]
    elems = bnd[:, 0] - 1
    nodes = bnd[:, 1:] - 1
    return CheartMeshSurface(tag, len(bnd), elems, nodes)
