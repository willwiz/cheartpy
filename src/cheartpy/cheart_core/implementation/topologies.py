import abc
import dataclasses as dc
from typing import Self, Text, TextIO

from cheartpy.cheart_core.pytools import join_fields

from ..aliases import *
from .basis import _CheartBasis
from ..interface import *




@dc.dataclass(slots=True)
class CheartTopology(_CheartTopology):
    name: str
    basis: _CheartBasis | None
    mesh: str
    fmt: VariableExportFormat = VariableExportFormat.TXT
    embedded: 'CheartTopology | None' = None
    partitioning_weight: int | None = None
    in_partition: bool = False
    continuous: bool = True
    spatial_constant: bool = False
    in_boundary: tuple["CheartTopology", int|str]|None = None

    def __repr__(self) -> str:
        return self.name

    def AddSetting(self, task: CheartTopologySetting, val: int | tuple[_CheartTopology, int] | None = None) -> None:
        match task, val:
            case _:
                raise ValueError(f"Setting for topology {self.name} {
                                 task} does not have a match value type")

    def create_in_boundary(self, top: "CheartTopology", surf: int|str) -> None:
        self.in_boundary = (top, surf)

    def write(self, f: TextIO):
        string = join_fields(
            self.name, self.mesh, self.basis if self.basis else "none")
        f.write(f"!DefTopology={{{string}}}\n")
        if self.embedded is not None:
            f.write(f"  !SetTopology={{{self.name}|EmbeddedInTopology|{self.embedded}}}\n")
        if self.in_boundary is not None:
            f.write(f"  !SetTopology={{{self.name}|CreateInBoundary|[{self.in_boundary[0]};{self.in_boundary[1]}]}}\n")


@dc.dataclass(slots=True)
class NullTopology(_CheartTopology):
    # method
    def __repr__(self) -> str:
        return "null_topology"

    def AddSetting(self, task: CheartTopologySetting, val: int | tuple[_CheartTopology, int] | None = None) -> None:
        raise ValueError("Cannot add setting to null topology")

    def write(self, f: TextIO):
        pass



@dc.dataclass(slots=True)
class TopInterface(_TopInterface):
    name: str
    method: TopologyInterfaceType
    topologies: list[_CheartTopology] = dc.field(default_factory=list)

    def write(self, f: TextIO):
        string = join_fields(self.method, *self.topologies)
        f.write(f"!DefInterface={{{string}}}\n")



@dc.dataclass(slots=True)
class OneToOneTopInterface(_TopInterface):
    name: str
    topologies: list[CheartTopology] = dc.field(default_factory=list)

    def write(self, f: TextIO):
        string = join_fields("OneToOne", *self.topologies)
        f.write(f"!DefInterface={{{string}}}\n")


@dc.dataclass(slots=True)
class ManyToOneTopInterface(_TopInterface):
    name: str
    topologies: list[CheartTopology]
    master_topology: CheartTopology
    interface_file: str
    nested_in_boundary: int|None = None

    def write(self, f: TextIO):
        nest_in_boundary =  None if self.nested_in_boundary is None else f"NestedInBndry[{self.nested_in_boundary}]"
        string = join_fields("ManyToOne", *self.topologies, self.master_topology, self.interface_file, nest_in_boundary)
        f.write(f"!DefInterface={{{string}}}\n")
