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

    def __repr__(self) -> str:
        return self.name

    def AddSetting(self, task: CheartTopologySetting, val: int | tuple[_CheartTopology, int] | None = None) -> None:
        match task, val:
            case _:
                raise ValueError(f"Setting for topology {self.name} {
                                 task} does not have a match value type")

    def write(self, f: TextIO):
        string = join_fields(
            self.name, self.mesh, self.basis if self.basis else "none")
        f.write(f"!DefTopology={{{string}}}\n")
        if self.embedded is not None:
            f.write(f"  !SetTopology={{{self.name}|EmbeddedInTopology|{self.embedded}}}\n")


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

