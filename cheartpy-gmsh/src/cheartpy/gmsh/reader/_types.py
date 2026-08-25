import dataclasses as dc
import enum
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.gmsh.tools import GmshBoundaryType
    from cheartpy.gmsh.types import Entity
    from cheartpy.mesh import CheartMesh, CheartMeshPatch
    from pytools.arrays import A1, A2


class BoundaryAssociation(enum.IntEnum):
    NONE = enum.auto()
    PARTIAL = enum.auto()
    FULL = enum.auto()


class GmshBndClass[I: np.integer](TypedDict, total=True):
    patch: CheartMeshPatch[I]
    kind: GmshBoundaryType


@dc.dataclass(slots=True)
class MultiDomainMesh[F: np.floating, I: np.integer]:
    volume: CheartMesh[F, I]
    subdomains: Mapping[int, A1[I]]
    boundaries: Mapping[int, Mapping[int, GmshBndClass[I]]]


class GmshTopInfo(NamedTuple):
    tag: Entity
    node_tags: A1[np.integer]
    elem_tags: A1[np.integer]
    connectivity: A2[np.integer]
    vol_type_id: int
    dim: int
