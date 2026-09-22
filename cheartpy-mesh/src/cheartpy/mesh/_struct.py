import dataclasses as dc
from typing import TYPE_CHECKING, Never, final

import numpy as np
from cheartpy.io import (
    check_for_meshes,
    chwrite_d_utf,
    chwrite_iarr_utf,
    chwrite_t_utf,
    fix_ch_sfx,
)

if TYPE_CHECKING:
    from collections.abc import ItemsView, Mapping, ValuesView
    from pathlib import Path

    from cheartpy.elem_interfaces import CheartEnum
    from pytools.arrays import A1, A2


__all__ = [
    "CheartMesh",
    "CheartMeshBoundary",
    "CheartMeshPatch",
    "CheartMeshSpace",
    "CheartMeshTopology",
]


@dc.dataclass(slots=True, frozen=True)
class CheartMeshSpace[F: np.floating = np.floating]:
    v: A2[F]

    @property
    def n(self) -> int:
        return len(self.v)

    def save(self, name: Path | str) -> None:
        chwrite_d_utf(name, self.v)


@dc.dataclass(slots=True, frozen=True)
class CheartMeshTopology[I: np.integer = np.integer, T: CheartEnum = CheartEnum]:
    v: A2[I]
    type: T

    @property
    def n(self) -> int:
        return len(self.v)

    def save(self, name: Path | str) -> None:
        chwrite_t_utf(name, self.v + 1, self.v.max() + 1)


@dc.dataclass(slots=True, frozen=True)
class CheartMeshPatch[I: np.integer = np.integer, B: CheartEnum = CheartEnum]:
    """Cheart Mesh Data for one face.

    tag: ToInt
        Label id for the patch
    n: ToInt
        No. of face elements in the patch
    k: A1[T]
        Array of element ids containing the corresponding face elements
    v: A2[T]
        Connectivity array of the face elements
    TYPE: VtkEnum
        VTK cell type of the face elements

    """

    tag: int
    k: A1[I]
    v: A2[I]
    type: B

    @property
    def n(self) -> int:
        return len(self.v)

    def to_array(self) -> A2[I]:
        res = np.pad(self.v + 1, ((0, 0), (1, 1)))
        res[:, 0] = self.k + 1
        res[:, -1] = self.tag
        return res


@final
@dc.dataclass(slots=True, frozen=True)
class CheartMeshBoundary[I: np.integer = np.integer, B: CheartEnum = CheartEnum]:
    v: Mapping[int, CheartMeshPatch[I, B]]

    def __bool__(self) -> bool:
        return bool(self.v)

    def __contains__(self, key: int) -> bool:
        return key in self.v

    @property
    def type(self) -> B | None:
        if not self.v:
            return None
        return next(iter(self.v.values())).type

    @property
    def n(self) -> int:
        return len(self.v)

    def items(self) -> ItemsView[int, CheartMeshPatch[I, B]]:
        return self.v.items()

    def values(self) -> ValuesView[CheartMeshPatch[I, B]]:
        return self.v.values()

    def save(self, name: Path | str) -> None:
        data = np.concatenate([v.to_array() for v in self.v.values()], axis=0)
        chwrite_iarr_utf(name, data)


@final
@dc.dataclass(slots=True, frozen=True, init=False)
class CheartMesh[
    F: np.floating = np.floating,
    I: np.integer = np.integer,
    T: CheartEnum = CheartEnum,
    B: CheartEnum = CheartEnum,
]:
    """Cheart Mesh Data.

    Attributes
    ----------
    space : CheartMeshSpace[F]
        The spatial data of the mesh.
    top : CheartMeshTopology[I]
        The topological data of the mesh.
    bnd : CheartMeshBoundary[I] | None
        The boundary data of the mesh, if it exists.

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
            The boundary data of the mesh, if it exists. If an empty dict is provided, it will be
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
        if check_for_meshes("prefix") and not forced:
            return
        prefix = fix_ch_sfx(prefix)
        self.space.save(f"{prefix}X")
        self.top.save(f"{prefix}T")
        if self.bnd:
            self.bnd.save(f"{prefix}B")
