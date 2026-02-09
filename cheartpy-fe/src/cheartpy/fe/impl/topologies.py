import dataclasses as dc
from typing import TYPE_CHECKING, Literal, TextIO

from cheartpy.fe.aliases import (
    CheartTopInterfaceEnum,
    CheartTopologyEnum,
    CheartTopologySetting,
    VariableExportEnum,
)
from cheartpy.fe.trait import ICheartBasis, ICheartTopology, ITopInterface
from cheartpy.fe.utils import get_enum, join_fields

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = [
    "CheartTopology",
    "ManyToOneTopInterface",
    "NullTopology",
    "OneToOneTopInterface",
    "TopInterface",
]


@dc.dataclass(slots=True)
class CheartTopology(ICheartTopology):
    name: str
    basis: ICheartBasis | None
    _mesh: Path
    fmt: VariableExportEnum = VariableExportEnum.TXT
    embedded: ICheartTopology | None = None
    partitioning_weight: int | None = None
    in_partition: bool = False
    continuous: bool = True
    spatial_constant: bool = False
    in_boundary: tuple[ICheartTopology, int | str] | None = None
    _discontinuous: bool = False

    def __repr__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        return True

    @property
    def mesh(self) -> Path:
        return self._mesh

    @property
    def order(self) -> Literal[0, 1, 2] | None:
        if self.basis is None:
            return None
        return self.basis.order

    @property
    def discontinuous(self) -> bool:
        return self._discontinuous

    @discontinuous.setter
    def discontinuous(self, val: bool) -> None:
        self._discontinuous = val

    def get_basis(self) -> ICheartBasis | None:
        return self.basis

    def add_setting(
        self,
        task: CheartTopologyEnum | CheartTopologySetting,
        val: int | ICheartTopology | tuple[ICheartTopology, int] | None = None,
    ) -> None:
        task = get_enum(task, CheartTopologyEnum)
        match task, val:
            case CheartTopologyEnum.EmbeddedInTopology, ICheartTopology() as topology:
                self.embedded = topology
            case _:
                msg = f"Setting for topology {self} {task} does not have a match value type"
                raise ValueError(msg)

    def create_in_boundary(self, top: ICheartTopology, surf: int | str) -> None:
        self.in_boundary = (top, surf)

    def write(self, f: TextIO) -> None:
        string = join_fields(self, self._mesh, self.basis or "none")
        f.write(f"!DefTopology={{{string}}}\n")
        if self.embedded is not None:
            f.write(f"  !SetTopology={{{self}|EmbeddedInTopology|{self.embedded}}}\n")
        if self.in_boundary is not None:
            line = (
                f"  !SetTopology={{{self}|CreateInBoundary|"
                f"[{self.in_boundary[0]};{self.in_boundary[1]}]}}\n"
            )
            f.write(line)
        if self._discontinuous:
            f.write(f"  !SetTopology={{{self}|MakeDiscontinuous}}\n")


@dc.dataclass(slots=True)
class NullTopology(ICheartTopology):
    # method
    def __repr__(self) -> str:
        return "null_topology"

    def __bool__(self) -> bool:
        return True

    @property
    def mesh(self) -> Path | None:
        return None

    @property
    def order(self) -> None:
        return None

    @property
    def discontinuous(self) -> bool:
        return self._discontinuous

    @discontinuous.setter
    def discontinuous(self, val: bool) -> None:
        self._discontinuous = val

    def get_basis(self) -> ICheartBasis | None:
        return None

    def add_setting(
        self,
        task: CheartTopologyEnum | CheartTopologySetting,
        val: int | ICheartTopology | tuple[ICheartTopology, int] | None = None,
    ) -> None:
        pass

    def write(self, f: TextIO) -> None:
        pass


@dc.dataclass(slots=True)
class TopInterface(ITopInterface):
    name: str
    _method: CheartTopInterfaceEnum
    topologies: list[ICheartTopology] = dc.field(default_factory=list[ICheartTopology])

    def write(self, f: TextIO) -> None:
        string = join_fields(self._method, *self.topologies)
        f.write(f"!DefInterface={{{string}}}\n")


@dc.dataclass(slots=True)
class OneToOneTopInterface(ITopInterface):
    name: str
    topologies: list[ICheartTopology] = dc.field(default_factory=list[ICheartTopology])

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash("_".join([str(s) for s in self.topologies]))

    @property
    def method(self) -> Literal["OneToOne"]:
        return "OneToOne"

    def get_master(self) -> ICheartTopology | None:
        return None

    def get_tops(self) -> Sequence[ICheartTopology]:
        return self.topologies

    def write(self, f: TextIO) -> None:
        string = join_fields("OneToOne", *self.topologies)
        f.write(f"!DefInterface={{{string}}}\n")


@dc.dataclass(slots=True)
class ManyToOneTopInterface(ITopInterface):
    name: str
    topologies: list[ICheartTopology]
    master_topology: ICheartTopology
    interface_file: Path
    nested_in_boundary: int | None = None

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(
            "_".join([str(s) for s in self.topologies]) + ":" + str(self.master_topology),
        )

    @property
    def method(self) -> Literal["ManyToOne"]:
        return "ManyToOne"

    def get_master(self) -> ICheartTopology | None:
        return self.master_topology

    def get_tops(self) -> Sequence[ICheartTopology]:
        return [self.master_topology, *self.topologies]

    def write(self, f: TextIO) -> None:
        nest_in_boundary = (
            None if self.nested_in_boundary is None else f"NestedInBndry[{self.nested_in_boundary}]"
        )
        string = join_fields(
            "ManyToOne",
            *self.topologies,
            self.master_topology,
            self.interface_file,
            nest_in_boundary,
        )
        f.write(f"!DefInterface={{{string}}}\n")
