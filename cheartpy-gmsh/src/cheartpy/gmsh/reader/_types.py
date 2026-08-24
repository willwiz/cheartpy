import dataclasses as dc
import enum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.gmsh.types import Entity
    from cheartpy.mesh import CheartMesh, CheartMeshBoundary
    from pytools.arrays import A1, A2


class BoundaryAssociation(enum.IntEnum):
    NONE = enum.auto()
    PARTIAL = enum.auto()
    FULL = enum.auto()


@dc.dataclass(slots=True)
class MultiDomainMesh[F: np.floating, I: np.integer]:
    volume: CheartMesh[F, I]
    subdomains: Mapping[int, A1[I]]
    boundaries: Mapping[int, CheartMeshBoundary[I] | None]


class GmshTopInfo(NamedTuple):
    tag: Entity
    node_tags: A1[np.integer]
    elem_tags: A1[np.integer]
    connectivity: A2[np.integer]
    vol_type_id: int
    dim: int
