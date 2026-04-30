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
    elem: str


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


@dc.dataclass(slots=True)
class CLPartition[F: np.floating]:
    prefix: str
    in_surf: int | None
    node: A1[F]
    domain: A2[F]

    def __repr__(self) -> str:
        return self.prefix

    def __str__(self) -> str:
        return self.prefix

    def astype[T: np.floating](self, dtype: DType[T]) -> CLPartition[T]:
        return CLPartition(
            prefix=self.prefix,
            in_surf=self.in_surf,
            node=self.node.astype(dtype),
            domain=self.domain.astype(dtype),
        )

    @property
    def dtype(self) -> DType[F]:
        return self.node.dtype


class CLMesh[F: np.floating, I: np.integer](NamedTuple):
    body: CheartMesh[F, I]
    iface: CheartMesh[F, I]
    domain: A2[F]
    elem: A2[F]
