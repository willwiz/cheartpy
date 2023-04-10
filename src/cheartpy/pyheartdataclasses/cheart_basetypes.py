#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, TextIO, Union, Type, Literal
from .cheart_pytools import *
from .cheart_terms import *


@dataclass
class TimeScheme:
  name: str
  start: int
  stop: int
  value: Union[int, str] # step or file
  def write(self, f:TextIO):
    f.write(hline("Define Time Scheme"))
    f.write(f'!DefTimeStepScheme={{{self.name}}}\n')
    f.write(f'  {self.start}  {self.stop}  {self.value}\n')

@dataclass
class Expression:
  name:str
  value:List[Union[str, float]]
  def __repr__(self) -> str:
    return self.name
  def write(self, f:TextIO):
    f.write(f'!DefExpression={{{self.name}}}\n')
    for v in self.value:
      f.write(f'  {v!s}\n')
    f.write('\n')


@dataclass
class Basis:
  name : str
  elem : Literal["POINT_ELEMENT", "point", "ONED_ELEMENT", "line", "QUADRILATERAL_ELEMENT",
                "quad", "TRIANGLE_ELEMENT", "tri", "HEXAHEDRAL_ELEMENT", "hex",
                "TETRAHEDRAL_ELEMENT", "tet"]
  type : Literal["NODAL_LAGRANGE#", "NL#", "MODAL_BASIS#", "PNODAL_BASIS#", "MINI_BASIS#",
                "NURBS_BASIS#", "SPECTRAL_BASIS#"]
  method : Literal['GAUSS_LEGENDRE#', 'GL#', 'KEAST_LYNESS#', 'KL#']
  def write(self, f:TextIO):
    f.write(f'!UseBasis={{{self.name}|{self.elem}|{self.type}|{self.method}}}\n')


@dataclass
class Topology:
  name : str
  mesh : str
  basis : Union[Basis,str]
  setting : List[List[Union[str, Topology]]] = field(default_factory=list)
  # methods
  def AddSetting(self, task:str, val):
    if isinstance(val, str):
      self.setting.append([task, val])
    elif isinstance(val, Topology):
      self.setting.append([task, val.name])
    else:
      raise TypeError('Type logic not implemented')
  def write(self, f:TextIO):
    f.write(
      (f'!DefTopology={{{self.name}|{self.mesh}'
       f'|{VoS(self.basis)}}}\n')
      )
    for s in self.setting:
      f.write(f'  !SetTopology={{{self.name}|{s[0]}|{VoS(s[1])}}}\n')

@dataclass
class TopInterface:
  name:str
  method : Literal["OneToOne", "ManyToOne"]
  topologies:List[Union[str,Topology]] = field(default_factory=list)
  def write(self, f:TextIO):
    f.write(f'!DefInterface={{{self.method}|{"|".join([(v) if isinstance(v, str) else v.name for v in self.topologies])}}}\n')

@dataclass
class Variable:
  name:str
  topology:Union[Topology, str]
  dim:int
  file:Optional[str] = None
  setting : List[List] = field(default_factory=list)
  def AddSetting(self,
      task:Literal["INIT_EXPR", "TEMPORAL_UPDATE_EXPR", "TEMPORAL_UPDATE_FILE",
        "ReadBinary", "ReadMMap"],
      val:Optional[Union[str,int,Variable,Expression]] = None):
    if val is None:
      self.setting.append([task])
    else:
      self.setting.append([task, val])
  def write(self, f:TextIO):
    if self.file is None:
      f.write(f'!DefVariablePointer={{{self.name}|{VoS(self.topology)}|{self.dim}}}\n')
    else:
      f.write(f'!DefVariablePointer={{{self.name}|{VoS(self.topology)}|{self.file}|{self.dim}}}\n')
    for s in self.setting:
      f.write(f'  !SetVariablePointer={{{self.name}|{s[0]}|{s[1]}}}\n')

@dataclass
class DataPointer:
  name:str
  file:str
  nt:int
  dim:int=2
  def write(self, f:TextIO):
    f.write(f'!DefDataPointer={{{self.name}|{self.file}|{self.dim}|{self.nt}}}\n')

@dataclass
class DataInterp:
  var:DataPointer
  def __repr__(self) -> str:
    return f'Interp({self.var.name},t)'