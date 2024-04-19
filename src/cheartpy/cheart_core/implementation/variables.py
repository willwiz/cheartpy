#!/usr/bin/env python3
import dataclasses as dc
from typing import Self, TextIO, Literal, overload

from .topologies import NullTopology
from ..aliases import *
from ..pytools import get_enum, join_fields
from ..interface import *
from .expressions import _Expression


@dc.dataclass(slots=True)
class Variable(_Variable):
    name: str
    topology: _CheartTopology
    dim: int
    data: str | None = None
    fmt: VariableExportFormat = VariableExportFormat.TXT
    freq: int = 1
    loop_step: int | None = None
    setting: tuple[VariableUpdateSetting, str | _Expression] | None = None
    expressions: dict[str, _Expression] = dc.field(default_factory=dict)

    def __repr__(self) -> str:
        return self.name

    def __getitem__[T:int | None](self, key: T) -> tuple[Self, T]:
        return (self, key)

    def idx(self, key: int) -> str:
        return f"{self.name}.{key}"

    @overload
    def AddSetting(
        self, task: Literal["INIT_EXPR", "TEMPORAL_UPDATE_EXPR"], val: _Expression) -> None: ...

    @overload
    def AddSetting(
        self, task: Literal["TEMPORAL_UPDATE_FILE", "TEMPORAL_UPDATE_FILE_LOOP"], val: str) -> None: ...

    @overload
    def AddSetting(
        self,
        task: Literal[
            "INIT_EXPR",
            "TEMPORAL_UPDATE_EXPR",
            "TEMPORAL_UPDATE_FILE",
            "TEMPORAL_UPDATE_FILE_LOOP",
        ],
        val: str | _Expression,
    ): ...

    def AddSetting(
        self,
        task: Literal[
            "INIT_EXPR",
            "TEMPORAL_UPDATE_EXPR",
            "TEMPORAL_UPDATE_FILE",
            "TEMPORAL_UPDATE_FILE_LOOP",
        ],
        val: str | _Expression,
    ):
        match task, val:
            case "INIT_EXPR" | "TEMPORAL_UPDATE_EXPR", _Expression():
                self.setting = (get_enum(task, VariableUpdateSetting), val)
            case "TEMPORAL_UPDATE_FILE", str():
                self.setting = (get_enum(task, VariableUpdateSetting), val)
            case "TEMPORAL_UPDATE_FILE_LOOP", str():
                self.setting = (get_enum(task, VariableUpdateSetting), val)
                if self.loop_step is None:
                    self.loop_step = self.freq
            case _:
                raise ValueError(f"Setting for variable {
                                 self.name} does not match correct type")
    def get_data(self)-> str|None:
        return self.data

    def get_top(self) -> list[_CheartTopology]:
        if isinstance(self.topology, NullTopology):
            return []
        return [self.topology]

    def get_expressions(
        self,
    ) -> list[_Expression]:

        if self.setting is None:
            expr = []
        elif isinstance(self.setting[1], str):
            expr = []
        else:
            expr = [self.setting[1]]
        return expr + [v for v in self.expressions.values()]

    def get_export_frequency(self) -> int:
        return self.freq

    def write(self, f: TextIO):
        string = join_fields(
            self.name, self.topology if self.topology else "null_topology", self.data, self.dim)
        f.write(
            f"!DefVariablePointer={{{string}}}\n"
        )
        if self.setting:
            string = join_fields(
                self.name, self.setting[0],
                self.setting[1], self.freq, self.loop_step
            )
            f.write(
                f"  !SetVariablePointer={{{string}}}\n"
            )
        if self.fmt == VariableExportFormat.BINARY or self.fmt == "BINARY":
            f.write(f"  !SetVariablePointer={{{self.name}|ReadBinary}}\n")
        elif self.fmt == VariableExportFormat.MMAP or self.fmt == "MMAP":
            f.write(f"  !SetVariablePointer={{{self.name}|ReadMMap}}\n")
