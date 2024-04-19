import abc
import dataclasses as dc
from typing import Self, Text, TextIO

from cheartpy.cheart_core.pytools import join_fields

from ..aliases import *
from .basis import CheartBasis


@dc.dataclass
class _CheartTopology(abc.ABC):

    # methods
    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...

    @abc.abstractmethod
    def AddSetting(self, task: CheartTopologySetting, val: int | tuple[Self, int] | None = None): ...


@dc.dataclass
class CheartTopology(_CheartTopology):
    name: str
    basis: CheartBasis | None
    mesh: str
    fmt: VariableExportFormat = VariableExportFormat.TXT
    embedded: 'CheartTopology | None' = None
    partitioning_weight: int | None = None
    in_partition: bool = False
    continuous: bool = True
    spatial_constant: bool = False

    def __repr__(self) -> str:
        return self.name

    def AddSetting(self, task: CheartTopologySetting, val: int | tuple[Self, int] | None = None) -> None:
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


@dc.dataclass
class NullTopology(_CheartTopology):
    # method
    def __repr__(self) -> str:
        return "null_topology"

    def AddSetting(self, task: CheartTopologySetting, val: int | tuple[Self, int] | None = None) -> None:
        raise ValueError("Cannot add setting to null topology")

    def write(self, f: TextIO):
        pass


def hash_tops(tops: list[_CheartTopology] | list[str]) -> str:
    names = [str(t) for t in tops]
    return "_".join(names)
