import dataclasses as dc
from typing import TYPE_CHECKING, NamedTuple, Required

import numpy as np
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from pathlib import Path

    from cheartpy.mesh import CheartMesh
    from pytools.arrays import A1, A2, DType


class APIKwargs(TypedDict, total=False):
    """Keyword arguments for CL API functions."""

    partition: CLPartition[np.floating]
    no_boundary: bool
    keep_left: bool


class CLPrefix(TypedDict, total=False):
    prefix: Required[str]
    body: str
    iface: str
    domain: str
    map: str


class CLVectorDef[F: np.floating](TypedDict, total=False):
    """Defn for centerline topology given vector of nodes."""

    home: Path
    prefix: Required[CLPrefix]
    in_surf: int
    a_z: Required[A1[F] | tuple[Path, DType[F]]]
    nodes: Required[A1[F]]


class CLSegmentDef[F: np.floating = np.float64](TypedDict, total=False):
    """Defn for centerline topology given segments."""

    home: Path
    prefix: Required[CLPrefix]
    in_surf: int
    a_z: Required[A1[F] | tuple[Path, DType[F]]]
    n: Required[int]


type CLDef[F: np.floating = np.float64] = CLVectorDef[F] | CLSegmentDef[F]


@dc.dataclass(slots=True, frozen=True)
class CLPartition[F: np.floating = np.float64, I: np.integer = np.intp]:
    nodes: A1[F]
    top: A2[I]
    domain: A2[F]

    @property
    def n(self) -> int:
        return len(self.nodes)

    @property
    def dtype(self) -> np.dtype[I]:
        return self.top.dtype

    @property
    def ftype(self) -> np.dtype[F]:
        return self.nodes.dtype

    def astype[V: np.floating, T: np.integer](
        self, ftype: DType[V], dtype: DType[T]
    ) -> CLPartition[V, T]:
        return CLPartition(
            nodes=np.astype(self.nodes, ftype),
            top=np.astype(self.top, dtype),
            domain=np.astype(self.domain, ftype),
        )


class CLMesh[F: np.floating, I: np.integer](NamedTuple):
    body: CheartMesh[F, I]
    iface: CheartMesh[F, I]
    domain: A2[F]
    elem: A2[F]
