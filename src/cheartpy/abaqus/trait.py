import enum
from collections.abc import Sequence
from typing import NamedTuple, Required, TypedDict


class _AbaqusElement(NamedTuple):
    tag: str
    nodes: Sequence[int]


class AbaqusElement(enum.Enum):
    S3R = _AbaqusElement("S3R", [0, 1, 2])
    T3D2 = _AbaqusElement("T3D2", [0, 1])
    T3D3 = _AbaqusElement("T3D3", [0, 1, 2])
    CPS3 = _AbaqusElement("CPS3", [0, 1, 2])
    CPS4 = _AbaqusElement("CPS4", [0, 1, 3, 2])
    CPS4_3D = _AbaqusElement("CPS4_3D", [0, 1, 3, 2, 4, 7, 8, 5, 6])
    C3D4 = _AbaqusElement("C3D4", [0, 1, 3, 2])
    TetQuad3D = _AbaqusElement("TetQuad3D", (0, 1, 3, 2, 4, 5, 7, 6))
    Tet3D = _AbaqusElement("Tet3D", [0, 1, 2, 3])


class AbaqusItem(enum.StrEnum):
    HEADING = "*heading"
    NODES = "*node"
    ELEMENTS = "*element"
    COMMENTS = "***"


class CMDInputKwargs(TypedDict, total=False):
    topology: Required[list[str]]
    prefix: str
    dim: int
    boundary: list[list[str]] | None
    masks: list[list[str]] | None
    cores: int
