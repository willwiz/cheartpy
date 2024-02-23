#!/usr/bin/env python3
from __future__ import annotations
import dataclasses as dc
from typing import TextIO, TypedDict, Union, Literal, overload
from .aliases import *
from .pytools import *
from .keywords import *


@dc.dataclass
class TimeScheme:
    name: str
    start: int
    stop: int
    value: int | str  # step or file

    def write(self, f: TextIO):
        f.write(hline("Define Time Scheme"))
        f.write(f"!DefTimeStepScheme={{{self.name}}}\n")
        f.write(f"  {self.start}  {self.stop}  {self.value}\n")


@dc.dataclass
class Expression:
    name: str
    value: list[str | float]

    def __repr__(self) -> str:
        return self.name

    def write(self, f: TextIO):
        f.write(f"!DefExpression={{{self.name}}}\n")
        for v in self.value:
            f.write(f"  {v!s}\n")
        f.write("\n")


@dc.dataclass(slots=True)
class Basis:
    basis: CheartBasisType
    order: int


@dc.dataclass(slots=True)
class Quadrature:
    elem: CheartQuadratureType
    gp: int


@dc.dataclass
class Setting:
    name: str
    value: list[str | int | float | Variable | Expression] | None = None

    def string(self) -> str:
        if self.value is None:
            return f"{self.name}"
        else:
            return f'{self.name}|{"|".join([VoS(v) for v in self.value])}'


@dc.dataclass
class CheartBasis:
    name: str
    elem: CheartElementType
    basis: Basis
    quadrature: Quadrature

    def write(self, f: TextIO):
        f.write(
            f"!UseBasis={{{self.name}|{self.elem}|{self.basis}{
                self.basis.order}|{self.quadrature}{self.quadrature.gp}}}\n"
        )


@dc.dataclass
class Topology:
    name: str
    mesh: str
    basis: CheartBasis | None
    setting: list[list[Union[str, Topology]]] = dc.field(default_factory=list)

    # methods
    def AddSetting(self, task: str, val):
        self.setting.append([task, val])

    def write(self, f: TextIO):
        f.write(
            (f"!DefTopology={{{self.name}|{self.mesh}" f"|{VoS(self.basis)}}}\n"))
        for s in self.setting:
            f.write(f"  !SetTopology={{{self.name}|{s[0]}|{VoS(s[1])}}}\n")


@dc.dataclass
class NullTopology:
    # method
    def __repr__(self) -> str:
        return "null_topology"

    def write(self, f: TextIO):
        pass


class EmbeddedTopology(Topology):
    def __init__(self, name: str, top: Topology, mesh: str | None = None) -> None:
        if mesh is None:
            m = top.mesh.split("_")[0] + "_emb"
        else:
            m = mesh
        super().__init__(name, m, None)
        self.AddSetting("EmbeddedInTopology", top)


@dc.dataclass
class TopInterface:
    name: str
    method: Literal["OneToOne", "ManyToOne"]
    topologies: list[Topology] = dc.field(default_factory=list)

    def write(self, f: TextIO):
        f.write(
            f'!DefInterface={{{self.method}|{"|".join(
                [(v) if isinstance(v, str) else v.name for v in self.topologies])}}}\n'
        )


class VariableSetting(TypedDict, total=False):
    INIT_EXPR: Expression
    TEMPORAL_UPDATE_EXPR: Expression
    TEMPORAL_UPDATE_FILE: str
    TEMPORAL_UPDATE_FILE_LOOP: str


@dc.dataclass
class Variable:
    name: str
    topology: Topology
    dim: int
    file: str | None = None
    format: VariableExportFormat = VariableExportFormat.TXT
    freq: int = 1
    loop_step: int | None = None
    setting: dict[str, Expression | str] = dc.field(default_factory=dict)

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
        self.setting[task] = val
        if (task == "TEMPORAL_UPDATE_FILE_LOOP") and (self.loop_step is None):
            self.loop_step = self.freq

    def write(self, f: TextIO):
        if self.file is None:
            f.write(
                f"!DefVariablePointer={{{self.name}|{
                    VoS(self.topology)}|{self.dim}}}\n"
            )
        else:
            f.write(
                f"!DefVariablePointer={{{self.name}|{VoS(self.topology)}|{
                    self.file}|{self.dim}}}\n"
            )
        for k, v in self.setting.items():
            if self.loop_step is None:
                f.write(
                    f"  !SetVariablePointer={{{self.name}|{
                        k}|{v}}}\n"
                )
            else:
                f.write(
                    f"  !SetVariablePointer={{{self.name}|{k}|{
                        v}|{self.freq}|{self.loop_step}}}\n"
                )
        if self.format == VariableExportFormat.BINARY or self.format == "BINARY":
            f.write(f"  !SetVariablePointer={{{self.name}|ReadBinary}}\n")
        elif self.format == VariableExportFormat.MMAP or self.format == "MMAP":
            f.write(f"  !SetVariablePointer={{{self.name}|ReadMMap}}\n")


@dc.dataclass
class DataPointer:
    name: str
    file: str
    nt: int
    dim: int = 2

    def write(self, f: TextIO):
        f.write(f"!DefDataPointer={{{self.name}|{
                self.file}|{self.dim}|{self.nt}}}\n")


@dc.dataclass
class DataInterp:
    var: DataPointer

    def __repr__(self) -> str:
        return f"Interp({self.var.name},t)"
