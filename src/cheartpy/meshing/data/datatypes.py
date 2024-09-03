import dataclasses as dc
import os
from ...io.cheartio import CHWrite_d_utf, CHWrite_iarr_utf, CHWrite_t_utf
import numpy as np
from ...types import Mat, Vec, f64, i32


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

    def save(self, name: str) -> None:
        data = np.concatenate([v.to_array() for v in self.v.values()], axis=0)
        CHWrite_iarr_utf(name, data)


@dc.dataclass(slots=True)
class CheartMesh:
    space: CheartMeshSpace
    top: CheartMeshTopology
    bnd: CheartMeshBoundary | None

    def save(self, prefix: str) -> None:
        self.space.save(f"{prefix}_FE.X")
        self.top.save(f"{prefix}_FE.T")
        if self.bnd is not None:
            self.bnd.save(f"{prefix}_FE.B")


def fix_suffix(prefix: str, suffix: str = "_FE.") -> str:
    for i in range(len(suffix), 0, -1):
        if prefix.endswith(suffix[:i]):
            return prefix + suffix[i:]
    return prefix + suffix


def create_bnd_surf(v: Mat[i32], tag: int) -> CheartMeshSurface:
    bnd = v[v[:, -1] == tag, :-1]
    elems = bnd[:, 0] - 1
    nodes = bnd[:, 1:] - 1
    return CheartMeshSurface(tag, len(bnd), elems, nodes)


def import_cheart_mesh(name: str) -> CheartMesh:
    prefix = fix_suffix(name)
    raw_space = np.loadtxt(f"{prefix}X", dtype=float, skiprows=1)
    raw_top = np.loadtxt(f"{prefix}T", dtype=int, skiprows=1) - 1
    space = CheartMeshSpace(len(raw_space), raw_space)
    top = CheartMeshTopology(len(raw_top), raw_top)
    if os.path.isfile(f"{prefix}B"):
        raw_bnd = np.loadtxt(f"{prefix}B", dtype=int, skiprows=1)
        bnd_tags = np.unique(raw_bnd[:, -1])
        bnd = CheartMeshBoundary(
            len(raw_bnd), {tag: create_bnd_surf(raw_bnd, tag) for tag in bnd_tags}
        )
    else:
        bnd = None
    return CheartMesh(space, top, bnd)
