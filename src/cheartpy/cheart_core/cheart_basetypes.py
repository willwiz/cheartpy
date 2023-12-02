#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, TextIO, Union, Type, Literal
from .cheart_alias import *
from .cheart_pytools import *
from .cheart_terms import *


@dataclass
class Setting:
    name: str
    value: Optional[List[str | int | float | Variable | Expression]] = None

    def string(self) -> str:
        if self.value is None:
            return f"{self.name}"
        else:
            return f'{self.name}|{"|".join([VoS(v) for v in self.value])}'


@dataclass
class TimeScheme:
    name: str
    start: int
    stop: int
    value: Union[int, str]  # step or file

    def write(self, f: TextIO):
        f.write(hline("Define Time Scheme"))
        f.write(f"!DefTimeStepScheme={{{self.name}}}\n")
        f.write(f"  {self.start}  {self.stop}  {self.value}\n")


@dataclass
class Expression:
    name: str
    value: List[Union[str, float]]

    def __repr__(self) -> str:
        return self.name

    def write(self, f: TextIO):
        f.write(f"!DefExpression={{{self.name}}}\n")
        for v in self.value:
            f.write(f"  {v!s}\n")
        f.write("\n")


@dataclass
class Basis:
    name: str
    elem: Literal[
        "POINT_ELEMENT",
        "point",
        "ONED_ELEMENT",
        "line",
        "QUADRILATERAL_ELEMENT",
        "quad",
        "TRIANGLE_ELEMENT",
        "tri",
        "HEXAHEDRAL_ELEMENT",
        "hex",
        "TETRAHEDRAL_ELEMENT",
        "tet",
    ]
    type: Literal[
        "NODAL_LAGRANGE#",
        "NL#",
        "MODAL_BASIS#",
        "PNODAL_BASIS#",
        "MINI_BASIS#",
        "NURBS_BASIS#",
        "SPECTRAL_BASIS#",
    ]
    method: Literal["GAUSS_LEGENDRE#", "GL#", "KEAST_LYNESS#", "KL#"]

    def write(self, f: TextIO):
        f.write(f"!UseBasis={{{self.name}|{self.elem}|{self.type}|{self.method}}}\n")


class HexahedralBasis(Basis):
    def __init__(
        self,
        name: str,
        type: Literal[
            "NODAL_LAGRANGE#",
            "NL#",
            "MODAL_BASIS#",
            "PNODAL_BASIS#",
            "MINI_BASIS#",
            "NURBS_BASIS#",
            "SPECTRAL_BASIS#",
        ],
        method: Literal["GAUSS_LEGENDRE#", "GL#"],
    ) -> None:
        Basis.__init__(self, name, "HEXAHEDRAL_ELEMENT", type, method)


class TETRAHEDRALBasis(Basis):
    def __init__(
        self,
        name: str,
        type: Literal[
            "NODAL_LAGRANGE#",
            "NL#",
            "MODAL_BASIS#",
            "PNODAL_BASIS#",
            "MINI_BASIS#",
            "NURBS_BASIS#",
            "SPECTRAL_BASIS#",
        ],
        method: Literal["GAUSS_LEGENDRE#", "GL#", "KEAST_LYNESS#", "KL#"],
    ) -> None:
        super().__init__(name, "TETRAHEDRAL_ELEMENT", type, method)


@dataclass
class Topology:
    name: str
    mesh: str
    basis: Optional[Basis]
    setting: List[List[Union[str, Topology]]] = field(default_factory=list)

    # methods
    def AddSetting(self, task: str, val):
        self.setting.append([task, val])
        # if isinstance(val, str):
        #   self.setting.append([task, val])
        # elif isinstance(val, Topology):
        #   self.setting.append([task, val.name])
        # else:
        #   raise TypeError('Type logic not implemented')

    def write(self, f: TextIO):
        f.write((f"!DefTopology={{{self.name}|{self.mesh}" f"|{VoS(self.basis)}}}\n"))
        for s in self.setting:
            f.write(f"  !SetTopology={{{self.name}|{s[0]}|{VoS(s[1])}}}\n")


class EmbeddedTopology(Topology):
    def __init__(self, name: str, top: Topology, mesh: Optional[str] = None) -> None:
        if mesh is None:
            m = top.mesh.split("_")[0] + "_emb"
        else:
            m = mesh
        super().__init__(name, m, "none")
        self.AddSetting("EmbeddedInTopology", top)


@dataclass
class TopInterface:
    name: str
    method: Literal["OneToOne", "ManyToOne"]
    topologies: List[Union[str, Topology]] = field(default_factory=list)

    def write(self, f: TextIO):
        f.write(
            f'!DefInterface={{{self.method}|{"|".join([(v) if isinstance(v, str) else v.name for v in self.topologies])}}}\n'
        )


@dataclass
class Variable:
    name: str
    topology: Union[Topology, str]
    dim: int
    file: Optional[str] = None
    format: VariableExportFormat | Literal[
        "TXT", "BINARY", "MMAP"
    ] = VariableExportFormat.TXT
    freq: int = 1
    loop_step: Optional[int] = None
    setting: List[Setting] = field(default_factory=list)

    def AddSetting(
        self,
        task: Literal[
            "INIT_EXPR",
            "TEMPORAL_UPDATE_EXPR",
            "TEMPORAL_UPDATE_FILE",
            "TEMPORAL_UPDATE_FILE_LOOP",
        ],
        val: Union[str, Expression],
    ):
        self.setting.append(Setting(task, [val]))
        if (task == "TEMPORAL_UPDATE_FILE_LOOP") and (self.loop_step is None):
            self.loop_step = self.freq

    def write(self, f: TextIO):
        if self.file is None:
            f.write(
                f"!DefVariablePointer={{{self.name}|{VoS(self.topology)}|{self.dim}}}\n"
            )
        else:
            f.write(
                f"!DefVariablePointer={{{self.name}|{VoS(self.topology)}|{self.file}|{self.dim}}}\n"
            )
        for s in self.setting:
            if self.loop_step is None:
                f.write(
                    f"  !SetVariablePointer={{{self.name}|{s.name}|{s.value[0]}}}\n"
                )
            else:
                f.write(
                    f"  !SetVariablePointer={{{self.name}|{s.name}|{s.value[0]}|{self.freq}|{self.loop_step}}}\n"
                )
        if self.format == VariableExportFormat.BINARY or self.format == "BINARY":
            f.write(f"  !SetVariablePointer={{{self.name}|ReadBinary}}\n")
        elif self.format == VariableExportFormat.MMAP or self.format == "MMAP":
            f.write(f"  !SetVariablePointer={{{self.name}|ReadMMap}}\n")


@dataclass
class DataPointer:
    name: str
    file: str
    nt: int
    dim: int = 2

    def write(self, f: TextIO):
        f.write(f"!DefDataPointer={{{self.name}|{self.file}|{self.dim}|{self.nt}}}\n")


@dataclass
class DataInterp:
    var: DataPointer

    def __repr__(self) -> str:
        return f"Interp({self.var.name},t)"
