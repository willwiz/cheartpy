import dataclasses as dc
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.io.api import (
    check_for_meshes,
    chwrite_d_utf,
    chwrite_iarr_utf,
    chwrite_t_utf,
    fix_ch_sfx,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from cheartpy.vtk.types import VtkType
    from pytools.arrays import A1, A2


__all__ = [
    "CheartMesh",
    "CheartMeshBoundary",
    "CheartMeshPatch",
    "CheartMeshSpace",
    "CheartMeshTopology",
]


@dc.dataclass(slots=True)
class CheartMeshSpace[T: np.floating]:
    n: int
    v: A2[T]

    def save(self, name: Path | str) -> None:
        chwrite_d_utf(name, self.v)


@dc.dataclass(slots=True)
class CheartMeshTopology[T: np.integer]:
    n: int
    v: A2[T]
    TYPE: VtkType

    def save(self, name: Path | str) -> None:
        chwrite_t_utf(name, self.v + 1, self.v.max() + 1)


@dc.dataclass(slots=True)
class CheartMeshPatch[T: np.integer]:
    tag: int
    n: int
    k: A1[T]
    v: A2[T]
    TYPE: VtkType

    def to_array(self) -> A2[T]:
        res = np.pad(self.v + 1, ((0, 0), (1, 1)))
        res[:, 0] = self.k + 1
        res[:, -1] = self.tag
        return res


@dc.dataclass(slots=True)
class CheartMeshBoundary[T: np.integer]:
    n: int
    v: Mapping[int, CheartMeshPatch[T]]
    TYPE: VtkType

    def save(self, name: Path | str) -> None:
        data = np.concatenate([v.to_array() for v in self.v.values()], axis=0)
        chwrite_iarr_utf(name, data)


@dc.dataclass(slots=True)
class CheartMesh[F: np.floating, I: np.integer]:
    space: CheartMeshSpace[F]
    top: CheartMeshTopology[I]
    bnd: CheartMeshBoundary[I] | None

    def save(self, prefix: Path | str, *, forced: bool = False) -> None:
        if check_for_meshes("prefix") and not forced:
            return
        prefix = fix_ch_sfx(prefix)
        self.space.save(f"{prefix}X")
        self.top.save(f"{prefix}T")
        if self.bnd is not None:
            self.bnd.save(f"{prefix}B")
