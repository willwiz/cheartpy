import dataclasses as dc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
type Tag = int
type Entity = int


@dc.dataclass(slots=True)
class GmshMeshTags:
    """Represents the tags for a Gmsh mesh."""

    dim: int
    domains: Mapping[Tag, Entity]
    boundarys: Mapping[Tag, tuple[Tag, Entity]]
