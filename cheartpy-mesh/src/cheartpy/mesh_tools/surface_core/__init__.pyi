from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from cheartpy.fe.aliases import TopologyDef
from cheartpy.vtk.types import VtkElem
from pytools.arrays import A2
from pytools.result import Result

from cheartpy.mesh import CheartMesh
from cheartpy.mesh_tools.tools import MergedMesh

type _COMPONENT = Literal["X", "Y", "Z"]
type _BOUND = (
    tuple[float, float]
    | tuple[_COMPONENT, float]
    | tuple[float, _COMPONENT]
    | tuple[_COMPONENT, _COMPONENT]
)
type _SURF_CONSTRAINTS = Mapping[_COMPONENT, _BOUND]

def create_mesh_from_surface[F: np.floating, I: np.integer](
    body: CheartMesh[F, I], surf_id: int
) -> Result[CheartMesh[F, I]]: ...
def normalize_by_row[F: np.floating](vals: A2[F]) -> A2[F]: ...
def compute_surface_normal_at_center[F: np.floating, I: np.integer](
    kind: VtkElem,
    space: A2[F],
    elem: A2[I],
) -> A2[F]: ...
def compute_surface_normal_at_nodes[F: np.floating, I: np.integer](
    kind: VtkElem,
    space: A2[F],
    elem: A2[I],
) -> dict[int, A2[F]]: ...
def compute_mesh_outer_normal_at_nodes[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> A2[F]: ...
def create_new_surface_in_mesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], constraints: _SURF_CONSTRAINTS, label: int
) -> Result[CheartMesh[F, I]]: ...
def create_new_surface_in_surf[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], surf_id: Iterable[int], constraints: _SURF_CONSTRAINTS, label: int
) -> Result[CheartMesh[F, I]]: ...
def compute_surface_normal[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], in_surf: int
) -> Result[A2[F]]: ...
def make_cutplane_topology[T](
    defn: Mapping[T, TopologyDef[T]],
    planes: Sequence[T],
    new_home: Path,
) -> Result[MergedMesh[np.float64, np.intp]]: ...
