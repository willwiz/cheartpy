__all__ = [
    "CheartMesh",
    "CheartMeshBoundary",
    "CheartMeshPatch",
    "CheartMeshSpace",
    "CheartMeshTopology",
]
import dataclasses as dc
from collections.abc import Mapping

import numpy as np
from arraystubs import Arr1, Arr2
from cheartpy.io.api import (
    check_for_meshes,
    chwrite_d_utf,
    chwrite_iarr_utf,
    chwrite_t_utf,
    fix_suffix,
)
from cheartpy.vtk.trait import VtkType


@dc.dataclass(slots=True)
class CheartMeshSpace[T: np.floating]:
    n: int
    v: Arr2[T]

    def save(self, name: str) -> None:
        chwrite_d_utf(name, self.v)


@dc.dataclass(slots=True)
class CheartMeshTopology[T: np.integer]:
    n: int
    v: Arr2[T]
    TYPE: VtkType

    def save(self, name: str) -> None:
        chwrite_t_utf(name, self.v + 1, self.v.max() + 1)


@dc.dataclass(slots=True)
class CheartMeshPatch[T: np.integer]:
    tag: int
    n: int
    k: Arr1[T]
    v: Arr2[T]
    TYPE: VtkType

    def to_array(self) -> Arr2[T]:
        res = np.pad(self.v + 1, ((0, 0), (1, 1)))
        res[:, 0] = self.k + 1
        res[:, -1] = self.tag
        return res


@dc.dataclass(slots=True)
class CheartMeshBoundary[T: np.integer]:
    n: int
    v: Mapping[int, CheartMeshPatch[T]]
    TYPE: VtkType

    def save(self, name: str) -> None:
        data = np.concatenate([v.to_array() for v in self.v.values()], axis=0)
        chwrite_iarr_utf(name, data)


@dc.dataclass(slots=True)
class CheartMesh[F: np.floating, I: np.integer]:
    space: CheartMeshSpace[F]
    top: CheartMeshTopology[I]
    bnd: CheartMeshBoundary[I] | None

    def save(self, prefix: str, *, forced: bool = False) -> None:
        if check_for_meshes("prefix") and not forced:
            return
        prefix = fix_suffix(prefix)
        self.space.save(f"{prefix}X")
        self.top.save(f"{prefix}T")
        if self.bnd is not None:
            self.bnd.save(f"{prefix}B")
