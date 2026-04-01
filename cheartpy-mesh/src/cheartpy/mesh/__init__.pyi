# ruff: noqa: PYI011
import dataclasses as dc
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from cheartpy.elem_interfaces import VtkEnum
from pytools.arrays import A1, A2, DType, ToInt
from pytools.result import Result

@dc.dataclass(slots=True)
class CheartMeshSpace[T: np.floating]:
    n: ToInt
    v: A2[T]
    def save(self, name: Path | str) -> None: ...

@dc.dataclass(slots=True)
class CheartMeshTopology[T: np.integer]:
    n: ToInt
    v: A2[T]
    TYPE: VtkEnum
    def save(self, name: Path | str) -> None: ...

@dc.dataclass(slots=True)
class CheartMeshPatch[T: np.integer]:
    tag: ToInt
    n: ToInt
    k: A1[T]
    v: A2[T]
    TYPE: VtkEnum
    def to_array(self) -> A2[T]: ...

@dc.dataclass(slots=True)
class CheartMeshBoundary[T: np.integer]:
    n: ToInt
    v: Mapping[int, CheartMeshPatch[T]]
    TYPE: VtkEnum
    def save(self, name: Path | str) -> None: ...

@dc.dataclass(slots=True)
class CheartMesh[F: np.floating, I: np.integer]:
    space: CheartMeshSpace[F]
    top: CheartMeshTopology[I]
    bnd: CheartMeshBoundary[I] | None
    def save(self, prefix: Path | str, *, forced: bool = False) -> None: ...

def import_cheart_mesh[F: np.floating, I: np.integer](
    name: Path | str,
    forced_type: VtkEnum | None = None,
    *,
    ftype: DType[F] = np.float64,
    itype: DType[I] = np.intp,
) -> Result[CheartMesh[F, I]]: ...
def remove_dangling_nodes[F: np.floating, I: np.integer](
    g: CheartMesh[F, I],
) -> CheartMesh[F, I]: ...
