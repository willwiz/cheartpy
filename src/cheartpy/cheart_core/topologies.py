import dataclasses as dc
from typing import Self, TextIO

from cheartpy.cheart_core.pytools import join_fields

from .aliases import *
from .bases import CheartBasis


@dc.dataclass
class _CheartTopology:
    name: str


@dc.dataclass
class CheartTopology(_CheartTopology):
    name: str
    basis: CheartBasis | None
    mesh: str
    fmt: VariableExportFormat = VariableExportFormat.TXT
    embedded: tuple[Self, int] | None = None
    partitioning_weight: int | None = None
    in_partition: bool = False
    continuous: bool = True
    spatial_constant: bool = False

    # methods
    def __repr__(self) -> str:
        return self.name

    def AddSetting(self, task: CheartTopologySetting, val: int | tuple[Self, int] | None = None):
        match task, val:
            case _:
                raise ValueError(f"Setting for topology {self.name} {
                                 task} does not have a match value type")

    def write(self, f: TextIO):
        string = join_fields(
            [self.name, self.mesh, self.basis if self.basis else "none"])
        f.write(f"!DefTopology={{{string}}}\n")


@dc.dataclass
class NullTopology(_CheartTopology):
    # method
    def __repr__(self) -> str:
        return "null_topology"

    def write(self, f: TextIO):
        pass


def hash_tops(tops: list[CheartTopology] | list[str]) -> str:
    names = [str(t) for t in tops]
    return "_".join(names)
