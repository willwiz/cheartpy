#!/usr/bin/env python3
import dataclasses as dc
from typing import Self, TextIO, Literal, overload
from .aliases import *
from .pytools import get_enum, join_fields
from .topologies import CheartTopology
from .expressions import Expression


@dc.dataclass
class Variable:
    name: str
    topology: CheartTopology | None
    dim: int
    file: str | None = None
    fmt: VariableExportFormat = VariableExportFormat.TXT
    freq: int = 1
    loop_step: int | None = None
    setting: tuple[VariableUpdateSetting, str | Expression] | None = None
    expressions: dict[str, Expression] = dc.field(default_factory=dict)

    def __repr__(self) -> str:
        return self.name

    def __getitem__[T:int | None](self, key: T) -> tuple[Self, T]:
        return (self, key)

    def idx(self, key: int) -> str:
        return f"{self.name}.{key}"

    @overload
    def AddSetting(
        self, task: Literal["INIT_EXPR", "TEMPORAL_UPDATE_EXPR"], val: Expression) -> None: ...

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
        val: str | Expression,
    ): ...

    def AddSetting(
        self,
        task: Literal[
            "INIT_EXPR",
            "TEMPORAL_UPDATE_EXPR",
            "TEMPORAL_UPDATE_FILE",
            "TEMPORAL_UPDATE_FILE_LOOP",
        ],
        val: str | Expression,
    ):
        match task, val:
            case "INIT_EXPR" | "TEMPORAL_UPDATE_EXPR", Expression():
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

    def write(self, f: TextIO):
        string = join_fields(
            self.name, self.topology if self.topology else "null_topology", self.file, self.dim)
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
