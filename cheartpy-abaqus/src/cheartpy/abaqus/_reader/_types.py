import dataclasses as dc
import enum
from typing import TYPE_CHECKING, Any, NamedTuple, Required, TypedDict, TypeIs

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from pytools.arrays import A1, DType


class _AbaqusElement(NamedTuple):
    tag: str
    nodes: Sequence[int]


class ElementEnum(enum.Enum, _AbaqusElement):
    T3D2 = _AbaqusElement("T3D2", [0, 1])
    T3D3 = _AbaqusElement("T3D3", [0, 1, 2])
    CPS3 = _AbaqusElement("CPS3", [0, 1, 2])
    CPS4 = _AbaqusElement("CPS4", [0, 1, 3, 2])
    CPS4_3D = _AbaqusElement("CPS4_3D", [0, 1, 3, 2, 4, 7, 8, 5, 6])
    C3D4 = _AbaqusElement("C3D4", [0, 1, 3, 2])
    S3R = _AbaqusElement("S3R", [0, 1, 2])
    TetQuad3D = _AbaqusElement("TetQuad3D", (0, 1, 3, 2, 4, 5, 7, 6))
    Tet3D = _AbaqusElement("Tet3D", [0, 1, 2, 3])


class AbaqusHeader(enum.StrEnum):
    HEADINGS = "*heading"
    NODES = "*node"
    ELEMENT = "*element"
    NSET = "*nset"
    ELSET = "*elset"
    COMMENTS = "***"


@dc.dataclass(slots=True)
class Content:
    type: AbaqusHeader
    next: str


def _is_mesh_elements(value: object) -> TypeIs[Element[np.integer[Any]]]:
    return isinstance(value, Element)


@dc.dataclass(slots=True)
class Headings:
    v: list[str]


@dc.dataclass(slots=True)
class Nodes[F: np.floating]:
    v: dict[int, A1[F]]


@dc.dataclass(slots=True)
class Element[I: np.integer]:
    name: str
    type: ElementEnum
    v: dict[int, A1[I]]

    def __hash__(self) -> int:
        return hash((self.name, self.type, len(self.v)))

    def __eq__(self, value: object) -> bool:
        if not _is_mesh_elements(value):
            return False
        if self.name != value.name or self.type != value.type:
            return False
        if len(self.v) != len(value.v):
            return False
        if self.v != value.v:
            return False
        return all(np.array_equal(self.v[k], value.v[k]) for k in self.v)


@dc.dataclass(slots=True)
class NSet[I: np.integer]:
    name: str
    v: A1[I]


@dc.dataclass(slots=True)
class ElSet[I: np.integer]:
    name: str
    v: A1[I]


@dc.dataclass(slots=True)
class AbaqusMesh[F: np.floating, I: np.integer]:
    headings: Headings
    nodes: Nodes[F]
    nset: dict[str, NSet[I]]
    elements: dict[str, Element[I]]
    elsets: dict[str, ElSet[I]]
    ftype: DType[F]
    dtype: DType[I]

    def add_headings(self, item: Headings) -> AbaqusMesh[F, I]:
        return AbaqusMesh(
            item, self.nodes, self.nset, self.elements, self.elsets, self.ftype, self.dtype
        )

    def add_nodes(self, item: Nodes[F]) -> AbaqusMesh[F, I]:
        nodes = Nodes(self.nodes.v | item.v)
        return AbaqusMesh(
            self.headings, nodes, self.nset, self.elements, self.elsets, self.ftype, self.dtype
        )

    def add_nset(self, item: NSet[I]) -> AbaqusMesh[F, I]:
        nset = self.nset | {item.name: item}
        return AbaqusMesh(
            self.headings, self.nodes, nset, self.elements, self.elsets, self.ftype, self.dtype
        )

    def add_element(self, item: Element[I]) -> AbaqusMesh[F, I]:
        elements = {item.name: item} | self.elements
        return AbaqusMesh(
            self.headings, self.nodes, self.nset, elements, self.elsets, self.ftype, self.dtype
        )

    def add_elset(self, item: ElSet[I]) -> AbaqusMesh[F, I]:
        elset = {item.name: item} | self.elsets
        return AbaqusMesh(
            self.headings, self.nodes, self.nset, self.elements, elset, self.ftype, self.dtype
        )


class CMDInputKwargs(TypedDict, total=False):
    topology: Required[list[str]]
    prefix: str
    dim: int
    boundary: list[list[str]] | None
    masks: list[list[str]] | None
    cores: int
