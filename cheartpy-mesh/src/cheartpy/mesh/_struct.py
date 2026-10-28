import bz2
import contextlib
import dataclasses as dc
import lzma
from pathlib import Path
from typing import TYPE_CHECKING, Never, Required, TypedDict

import numpy as np
from cheartpy.elem_interfaces import CheartEnum
from cheartpy.io import (
    check_for_meshes,
    chwrite,
    fix_ch_sfx,
)

if TYPE_CHECKING:
    from collections.abc import ItemsView, Mapping, ValuesView

    from pytools.arrays import A1, A2


__all__ = [
    "CheartMesh",
    "CheartMeshBoundary",
    "CheartMeshPatch",
    "CheartMeshSpace",
    "CheartMeshTopology",
]


@dc.dataclass(slots=True, frozen=True, init=False)
class CheartMeshSpace[F: np.floating = np.floating]:
    """Spatial coordinate part of Cheart Mesh.

    Type Parameters
    ---------------
    F : np.floating

    Attributes
    ----------
    v : A2[F]
        Array of spatial coordinates of the mesh nodes.
    n : int
        Number of nodes in the mesh.

    Methods
    -------
    save(name: Path | str)
        Saves the spatial coordinates to a file with the given name.

    """

    v: A2[F]

    def __init__(self, v: A2[F]) -> None:
        object.__setattr__(self, "v", v)

    @property
    def n(self) -> int:
        """Number of nodes in the mesh."""
        return len(self.v)

    def save(self, name: Path | str) -> None:
        chwrite(name, self.v, fmt="X")


@dc.dataclass(slots=True, frozen=True, init=False)
class CheartMeshTopology[I: np.integer = np.integer, T: CheartEnum = CheartEnum]:
    """Topology nodal connectivity of elements.

    Type Parameters
    ---------------
    I : np.integer
    T : CheartEnum

    Attributes
    ----------
    v : A2[I]
        Array of element connectivity, where each row corresponds to an element and contains the
        indices of its nodes in order corresponding to `kind`.
    n : int
        Number of elements in the mesh.
    kind : T
        The type of elements in the mesh, represented as a CheartEnum.

    Methods
    -------
    to_array() -> A2[I]
        Returns the connectivity array with 1-based indexing.
    save(name: Path | str)
        Saves the connectivity array to a file with the given name, using 1-based indexing and
        the maximum node index plus one as the number of nodes.

    """

    v: A2[I]
    kind: T

    def __init__(self, v: A2[I], kind: T) -> None:
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "kind", kind)

    @property
    def n(self) -> int:
        return len(self.v)

    def to_array(self) -> A2[I]:
        return self.v + 1

    def save(self, name: Path | str) -> None:
        chwrite(name, self.to_array(), fmt="T")


@dc.dataclass(slots=True, frozen=True, init=False)
class CheartMeshPatch[I: np.integer = np.integer, B: CheartEnum = CheartEnum]:
    """Cheart Mesh data for one surface patch.

    Type Parameters
    ---------------
    I : np.integer
    B : CheartEnum

    Attributes
    ----------
    tag: int
        Label id for the patch
    n: int
        No. of face elements in the patch
    k: A1[B]
        Array of element ids containing the corresponding face elements
    v: A2[B]
        Connectivity array of the face elements
    kind: VtkEnum
        VTK cell type of the face elements

    Methods
    -------
    to_array() -> A2[I]
        Returns the patch data as a 2D array with 1-based indexing, where each row corresponds to a
        face element and contains the element id, the node indices, and the tag.

    """

    tag: int
    k: A1[I]
    v: A2[I]
    kind: B

    def __init__(self, tag: int, k: A1[I], v: A2[I], kind: B) -> None:
        object.__setattr__(self, "tag", tag)
        object.__setattr__(self, "k", k)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "kind", kind)

    @property
    def n(self) -> int:
        return len(self.v)

    def to_array(self) -> A2[I]:
        res = np.pad(self.v + 1, ((0, 0), (1, 1)))
        res[:, 0] = self.k + 1
        res[:, -1] = self.tag
        return res


@dc.dataclass(slots=True, frozen=True, init=False)
class CheartMeshBoundary[I: np.integer = np.integer, B: CheartEnum = CheartEnum]:
    """Cheart Mesh Boundary Data Structure.

    Hold all boundary surface data of the mesh.

    Type Parameters
    ---------------
    I : np.integer
    B : CheartEnum
        Boundary element type.

    Attributes
    ----------
    v : Mapping[int, CheartMeshPatch[I, B]]
        A mapping from surface index to CheartMeshPatch instances for the surface.
    n : int
        Number of boundary surfaces in the mesh.
    kind : B | None
        The type of boundary elements in the mesh, represented as a CheartEnum. Returns None if
        there are no boundary surfaces.

    Methods
    -------
    items() -> ItemsView[int, CheartMeshPatch[I, B]]
        Returns a view of the items in the boundary mapping.
    values() -> ValuesView[CheartMeshPatch[I, B]]
        Returns a view of the values in the boundary mapping.
    save(name: Path | str)
        Saves the boundary data to a file with the given name, concatenating all patches into a
        single array and using 1-based indexing.

    """

    v: Mapping[int, CheartMeshPatch[I, B]]

    def __init__(self, v: Mapping[int, CheartMeshPatch[I, B]]) -> None:
        object.__setattr__(self, "v", v)

    def __bool__(self) -> bool:
        return bool(self.v)

    def __contains__(self, key: int) -> bool:
        return key in self.v

    @property
    def kind(self) -> B | None:
        if not self.v:
            return None
        return next(iter(self.v.values())).kind

    @property
    def n(self) -> int:
        return len(self.v)

    def items(self) -> ItemsView[int, CheartMeshPatch[I, B]]:
        return self.v.items()

    def values(self) -> ValuesView[CheartMeshPatch[I, B]]:
        return self.v.values()

    def save(self, name: Path | str) -> None:
        data = np.concatenate([v.to_array() for v in self.v.values()], axis=0)
        chwrite(name, data, fmt="B")


@dc.dataclass(slots=True, frozen=True, init=False)
class CheartMesh[
    F: np.floating = np.floating,
    I: np.integer = np.integer,
    T: CheartEnum = CheartEnum,
    B: CheartEnum = CheartEnum,
]:
    """Cheart Mesh Data Structure.

    Type Parameters
    ---------------
    F : np.floating
        Floating point type.
    I : np.integer
        Integer type.
    T : CheartEnum
        Enum type for mesh element.
    B : CheartEnum
        Enum type for boundary element.

    Attributes
    ----------
    space : CheartMeshSpace[F]
        The spatial data of the mesh.
    top : CheartMeshTopology[I, T]
        The topological data of the mesh.
    bnd : CheartMeshBoundary[I, B]
        The boundary data of the mesh, if it exists.

    Methods
    -------
    save(prefix: Path | str, *, forced: bool = False)
        Save the Cheart mesh data to files with the given prefix. If `forced` is True, overwrite
        existing files. If the prefix ends with ".npz", ".npz.xz

    """

    space: CheartMeshSpace[F]
    top: CheartMeshTopology[I, T]
    bnd: CheartMeshBoundary[I, B]

    def __init__(
        self,
        space: CheartMeshSpace[F],
        top: CheartMeshTopology[I, T],
        bnd: CheartMeshBoundary[I, B] | Mapping[Never, Never],
    ) -> None:
        """Initialize a CheartMesh instance.

        Parameters
        ----------
        space : CheartMeshSpace[F]
            The spatial data of the mesh.
        top : CheartMeshTopology[I, T]
            The topological data of the mesh.
        bnd : CheartMeshBoundary[I, B] | Mapping[Never, Never]
            The boundary data of the mesh. If an empty dict is provided, it will be
            converted to a CheartMeshBoundary instance with no patches.

        """
        object.__setattr__(self, "space", space)
        object.__setattr__(self, "top", top)
        object.__setattr__(self, "bnd", bnd or CheartMeshBoundary({}))

    def save(self, prefix: Path | str, *, forced: bool = False) -> None:
        """Save the Cheart mesh data to files with the given prefix.

        Parameters
        ----------
        prefix : Path | str
            The prefix for the output files.
        forced : bool, optional
            If True, overwrite existing files. Default is False.

        """
        prefix = Path(prefix)
        if prefix.name.endswith((".npz", ".npz.xz", ".npz.bz2")):
            _export_cheartmesh_compressed(self, prefix, forced=forced)
            return
        _export_cheartmesh_cheart(self, prefix, forced=forced)


def _export_cheartmesh_cheart[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], prefix: Path | str, *, forced: bool = False
) -> None:
    if check_for_meshes(str(prefix)) and not forced:
        return
    prefix = fix_ch_sfx(prefix)
    mesh.space.save(f"{prefix}X")
    mesh.top.save(f"{prefix}T")
    if mesh.bnd:
        mesh.bnd.save(f"{prefix}B")


class CheartMeshCompressed[F: np.floating, I: np.integer](TypedDict, total=False, closed=True):
    """Compressed representation of a CheartMesh instance.

    Attributes
    ----------
    x : Required[A2[F]]
        The spatial data of the mesh.
    t : Required[A2[I]]
        The topological data of the mesh.
    b : A2[I]
        The boundary data of the mesh, if it exists.

    """

    x: Required[A2[F]]
    t: Required[A2[I]]
    b: A2[I]


def archive_cheartmesh[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I],
) -> CheartMeshCompressed[F, I]:
    """Archive a CheartMesh instance into a CheartMeshCompressed dictionary.

    Parameters
    ----------
    mesh : CheartMesh[F, I]
        The CheartMesh instance to archive.

    Returns
    -------
    CheartMeshCompressed[F, I]
        A dictionary containing the archived mesh data.

    """
    compressed_mesh: CheartMeshCompressed[F, I] = {"x": mesh.space.v, "t": mesh.top.to_array()}
    if mesh.bnd:
        compressed_mesh["b"] = np.concatenate([v.to_array() for v in mesh.bnd.values()], axis=0)
    return compressed_mesh


_ARCHIVE_LIB = {
    ".xz": lzma,
    ".bz2": bz2,
    ".npz": contextlib,
}


def _export_cheartmesh_compressed[F: np.floating, I: np.integer](
    mesh: CheartMesh[F, I], prefix: Path, *, forced: bool = False
) -> None:
    if prefix.exists() and not forced:
        return
    mesh_archive = archive_cheartmesh(mesh)
    if prefix.name.endswith(".npz.xz"):
        with lzma.open(prefix, "wb") as f:
            np.savez(f, **mesh_archive)
    elif prefix.name.endswith(".npz.bz2"):
        with bz2.open(prefix, "wb") as f:
            np.savez(f, **mesh_archive)
    else:
        np.savez_compressed(prefix, **mesh_archive)
