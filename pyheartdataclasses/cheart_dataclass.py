#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import List, Optional, Dict, TextIO, Union, Type
from .cheart_terms import *
from .cheart_pytools import *
from .cheart_basetypes import *
from .cheart_dictionary import *


"""
Cheart dataclasses

Structure:

PFile : {TimeScheme, SolverGroup, SolverMatrix, Basis, Topology, TopInterface,
         Variable, Problem, Expressions}

Topology -> TopInterface

BCPatch -> BoundaryCondtion
Matlaw -> SolidProblem(Problem)
BoundaryCondtion -> Problem


TimeScheme -> SolverGroup
SolverSubgroup -> SolverGroup
Problem -> SolverMatrix -> SolverSubgroup
Problem -> SolverSubgroup (method: SOLVER_SEQUENTIAL)


Content:

TimeScheme
Basis
Topology
TopInterface
Variable
BCPatch
BoundaryCondtion
Matlaw (SolidProblem)
Problem : {SolidProblem}
SolverMatrix
SolverSubgroup
SolverGroup
Expressions
PFile
"""

@dataclass
class BCPatch:
  id:Union[int, str]
  component:str
  type:Literal['dirichlet','neumann','neumann_ref','neumann_nl','stabilized_neumann','consistent']
  value:Union[Expression,str,int]
  options:List[Union[str,int,float]] = field(default_factory=list)
  def print(self):
    return f'    {self.id}  {VoS(self.component)}  {self.type}  {VoS(self.value)}  {"  ".join(self.options)}\n'


@dataclass
class BoundaryCondition:
  patches:Optional[List[BCPatch]] = None
  type:Optional[str] = None
  def AddPatch(self, *patch:BCPatch):
    if self.patches is None:
      self.patches = list()
    for p in patch:
      self.patches.append(p)
  def DefPatch(self, id:Union[int, str], component:str, type:str, val:Union[str,int]):
    self.patches.append(BCPatch(id, component, type, val))
  def write(self, f:TextIO):
    if self.patches is None:
      f.write(f'  !Boundary-conditions-not-required\n\n')
    else:
      f.write(f'  !Boundary-patch-definitions\n')
      for p in self.patches:
        f.write(p.print())
      f.write('\n')

# Matlaws -----------------------------------------------------------------------------
@dataclass
class Law:
  pass

@dataclass
class Matlaw(Law):
  name:str
  parameters:List[float]
  def print(self):
    return (f'  !ConstitutiveLaw={{{self.name}}}\n'
            f'    {"  ".join([str(i) for i in self.parameters])}\n'
    )

@dataclass()
class FractionalVE(Law):
  alpha:float
  np:int
  Tf:float
  store:Union[Variable,str]
  Tscale:Optional[float]=10.0
  name:str="fractional-ve"
  InitPK2:Union[bool,int]=False
  ZeroPK2:bool=False
  Order:Literal[1,2]=2
  laws:List[Matlaw]=field(default_factory=list)
  def AddLaw(self, *law:Type[Law]):
    for v in law:
      self.laws.append(v)
  def print(self):
    l = (
      f'  !ConstitutiveLaw={{{self.name}}}\n'
      f'    {VoS(self.store)}\n'
      f'    {self.alpha}  {self.np}  {self.Tf}  {"" if self.Tscale is None else self.Tscale}\n'
    )
    if self.InitPK2:
      l = l + f'    InitPK2  {self.InitPK2 if (type(self.InitPK2) is int) else ""}\n'
    if self.ZeroPK2:
      l = l + '    ZeroPK2\n'
    if self.Order != 2:
      l = l + '    Order 1\n'
    for v in self.laws:
      l = l +f'    {v.name}  [{" ".join([str(i) for i in v.parameters])}]\n'
    return l

@dataclass()
class FractionalDiffEQ(Law):
  alpha   : float
  delta   : float
  np      : int
  Tf      : float
  store   : Union[Variable,str]
  Tscale  : Optional[float] = 10.0
  name    : str             = "fractional-diffeq"
  InitPK2 : Union[bool,int] = False
  ZeroPK2 : bool            = False
  Order   : Literal[1,2]    = 2
  laws    : List[Matlaw]=field(default_factory=list)
  def AddLaw(self, *law:Type[Law]):
    for v in law:
      self.laws.append(v)
  def print(self):
    l = (
      f'  !ConstitutiveLaw={{{self.name}}}\n'
      f'    {VoS(self.store)}\n'
      f'    {self.alpha}  {self.np}  {self.Tf}  {self.delta}  {"" if self.Tscale is None else self.Tscale}\n'
    )
    if self.InitPK2:
      l = l + f'    InitPK2  {self.InitPK2 if (type(self.InitPK2) is int) else ""}\n'
    if self.ZeroPK2:
      l = l + '    ZeroPK2\n'
    if self.Order != 2:
      l = l + '    Order 1\n'
    counter = 0
    for v in self.laws:
      if isinstance(v,Matlaw):
        l = l +f'    HE  law  {v.name}  [{" ".join([str(i) for i in v.parameters])}]\n'
      elif isinstance(v,FractionalVE):
        counter = counter + 1
        sc =  "" if v.Tscale is None else v.Tscale
        l = l +f'    frac{counter}  parm  {VoS(v.store)}  {v.alpha}  {v.np}  {v.Tf}  {sc}\n'
        for law in v.laws:
          l = l +f'    frac{counter}  law   {law.name}  [{" ".join([str(i) for i in law.parameters])}]\n'
    return l

# Problems ----------------------------------------------------------------------

@dataclass
class Problem:
  name: str
  problem: str
  vars: Dict[str, Variable] = field(default_factory=dict)
  options: Dict[str, List[str]] = field(default_factory=dict)
  flags: List[str] = field(default_factory=list)
  BC: BoundaryCondition = field(default_factory=BoundaryCondition)
  def UseVariable(self, req:str, var:Variable) -> None:
    self.vars[req] = var
  def UseOption(self, opt:str, *val:str) -> None:
    if val:
      self.options[opt] = list(val)
    else:
      self.flags.append(opt)
  def write(self, f:TextIO):
    f.write(f'!DefProblem={{{self.name}|{self.problem}}}\n')
    for k, v in self.vars.items():
      f.write(f'  !UseVariablePointer={{{k}|{v.name}}}\n')
    for k, v in self.options.items():
      f.write(f'  !{k}={{{"|".join([str(VoS(i)) for i in v])}}}\n')
    for v in self.flags:
      f.write(f'  !{v}\n')

@dataclass
class SolidProblem(Problem):
  problem: str = 'quasi_static_elasticity'
  matlaws: List[Matlaw] = field(default_factory=list)
  def AddMatlaw(self, *law:Matlaw):
    for v in law:
      self.matlaws.append(v)
  def write(self, f: TextIO):
    super().write(f)
    for v in self.matlaws:
      f.write(v.print())
    self.BC.write(f)

@dataclass
class L2Projection(Problem):
  problem: str = 'l2solidprojection_problem'
  def UseVariable(self, req: Literal['Space', 'Variable'],
      var: Variable) -> None:
    return super().UseVariable(req, var)
  def UseOption(self, opt: Literal['Mechanical-Problem','Projected-Variable', 'Solid-Master-Override'],
      *val: str) -> None:
    return super().UseOption(opt, *val)
  def write(self, f: TextIO):
    super().write(f)
    self.BC.write(f)

@dataclass
class NormProblem(Problem):
  problem: str = 'norm_calculation'
  def UseVariable(self, req: Literal['Space', 'Term1', 'Term2'], var: Variable) -> None:
    return super().UseVariable(req, var)
  def UseOption(self, opt: Literal['Boundary-normal','Output-filename'], *val: str) -> None:
    return super().UseOption(opt, *val)
  def write(self, f: TextIO):
    super().write(f)
    self.BC.write(f)

# Solver Matrix
@dataclass
class SolverMatrix:
  name: str
  method: str
  problem: List[Type[Problem]] = field(default_factory=list)
  settings: Optional[Dict[str,List]] = field(default_factory=dict)
  def AddSetting(self, opt, *val):
    self.settings[opt] = list(val)
  def write(self, f:TextIO):
    f.write(f'!DefSolverMatrix={{{self.name}|{self.method}|{"|".join([p.name for p in self.problem])}}}\n')
    for k,v in self.settings.items():
      if v:
        f.write(f' !SetSolverMatrix={{{self.name}|{k}|{"|".join(v)}}}\n')
      else:
        f.write(f' !SetSolverMatrix={{{self.name}|{k}}}\n')


# Define Solver SubGroup
@dataclass
class SolverSubGroup:
  name : str = field(default='none')
  method : Literal["seq_fp_linesearch","SOLVER_SEQUENTIAL"] = field(default='seq_fp_linesearch')
  problems:List[Union[SolverMatrix,Type[Problem],str]] = field(default_factory=list)



@dataclass
class SolverGroup(object):
  name : str
  time : TimeScheme
  SolverSubGroups : List[SolverSubGroup] = field(default_factory=list)
  aux_vars : Dict[str, Union[str, Variable]] = field(default_factory=dict)
  export_initial_condition : bool = False
  settings : Dict[str, Union[str, float]] = field(default_factory=dict)
  # TOL
  def AddSetting(self,
    setting:Literal["L2TOL", "L2PERCENT", "INFRES", "INFUPDATE", "INFDEL",
                    "ITERATION", "SUBITERATION", "LINESEARCHITER", "SUBITERFRACTION",
                    "INFRELUPDATE", "L2RESRELPERCENT"],
    val:Union[Expression,Variable,float,str]) -> None:
    self.settings[setting] = val
  # VAR
  def AddVariable(self, *var:Union[str, Variable]):
    for v in var:
      if isinstance(v, str):
        self.aux_vars[v] = v
      else:
        self.aux_vars[v.name] = v
  def RemoveVariable(self, *var:Union[str, Variable]):
    for v in var:
      if isinstance(v, str):
        self.aux_vars.pop(v)
      else:
        self.aux_vars.pop(v.name)
  # SG
  def AddSolverSubGroup(self, *sg:SolverSubGroup) -> None:
    for v in sg:
      self.SolverSubGroups.append(v)
  def RemoveSolverSubGroup(self, *sg:SolverSubGroup) -> None:
    for v in sg:
      self.SolverSubGroups.remove(v.name)
  def MakeSolverSubGroup(self, method:Literal["seq_fp_linesearch","SOLVER_SEQUENTIAL"],
      *problems:Union[SolverMatrix,Type[Problem],str]) -> None:
    self.SolverSubGroups.append(SolverSubGroup(method=method, problems=list(problems)))
  # WRITE
  def write(self, f:TextIO) -> None:
    if isinstance(self.time,TimeScheme):
      self.time.write(f)
    f.write(hline("Solver Groups"))
    f.write(f'!DefSolverGroup={{{self.name}|{VoS(self.time)}}}\n')
    # Handle Additional Vars
    vars = [VoS(v) for v in self.aux_vars.values()]
    for l in splicegen(45, vars):
      if l:
        f.write(f'  !SetSolverGroup={{{self.name}|AddVariables|{"|".join(l)}}}\n')
    # Print export init setting
    if self.export_initial_condition:
      f.write(f'  !SetSolverGroup={{{self.name}|export_initial_condition}}\n')
    # Print Conv Settings
    for key, val in self.settings.items():
      f.write(f'  !SetSolverGroup={{{self.name}|{key}|{val!s}}}\n')
    for g in self.SolverSubGroups:
      pobs = [VoS(p) for p in g.problems]
      f.write(f'!DefSolverSubGroup={{{self.name}|{g.method}|{"|".join(pobs)}}}\n')





@dataclass
class PFile(object):
  h     : str = ""
  times      : Dict[str, TimeScheme]    = field(default_factory=dict)
  solverGs   : Dict[str, SolverGroup]   = field(default_factory=dict)
  matrices   : Dict[str, SolverMatrix]  = field(default_factory=dict)
  bases      : Dict[str, Basis]         = field(default_factory=dict)
  toplogies  : Dict[str, Topology]      = field(default_factory=dict)
  interfaces : Dict[str, TopInterface]  = field(default_factory=dict)
  vars       : Dict[str, Variable]      = field(default_factory=dict)
  dataPs     : Dict[str, DataPointer]   = field(default_factory=dict)
  problems   : Dict[str, Type[Problem]] = field(default_factory=dict)
  exprs      : Dict[str, Variable]      = field(default_factory=dict)
  output_path : Optional[str] = None
  exportfrequencies: Dict[List,str] = field(default_factory=dict)
  def SetOutputPath(self, path):
    self.output_path = path
  # Add Time Scheme
  def AddTimeScheme(self, *time:TimeScheme) -> None:
    for t in time:
      self.times[t.name] = t
  # Add Basis
  def AddBasis(self, *basis:Basis) -> None:
    for b in basis:
      self.bases[b.name] = b
  # Add Topology
  def AddTopology(self, *top:Topology) -> None:
    for t in top:
      self.toplogies[t.name] = t
  def SetTopology(self, name, task, val) -> None:
    self.toplogies[name].AddSetting(task, val)
  # Add Interfaces
  def AddInterface(self, *interface:TopInterface) -> None:
    for v in interface:
      self.interfaces[v.name] = v
  # Add Variables
  def AddVariable(self, *var:Variable) -> None:
    for v in var:
      self.vars[v.name] = v
  def SetVariable(self, name, task, val:Optional[Union[str,int]]) -> None:
    self.vars[name].AddSetting(task, val)
  # Add Data Pointers
  def AddDataPointer(self, *var:Variable) -> None:
    for v in var:
      self.dataPs[v.name] = v
  # Set Export Frequency
  def SetExportFrequency(self, *vars :Union[Variable,str], freq:int = 1):
    if str(freq) in self.exportfrequencies:
      self.exportfrequencies[str(freq)].extend([VoS(v) for v in vars])
    else:
      self.exportfrequencies[str(freq)] = [VoS(v) for v in vars]
  # Problem
  def AddProblem(self, *prob:Problem) -> None:
    for v in prob:
      self.problems[v.name] = v
  # Matrix
  def AddMatrix(self, *mat:SolverMatrix) -> None:
    for v in mat:
      self.matrices[v.name] = v
  # SolverGroup
  def AddSolverGroup(self, *grp:SolverGroup) -> None:
    for v in grp:
      self.solverGs[v.name] = v
  # Expression
  def AddExpression(self, *expr:Expression) -> None:
    for v in expr:
      self.exprs[v.name] = v


  # ----------------------------------------------------------------------------
  # Producing the Pfile
  def write(self, f:TextIO):
    f.write(header(self.h))
    f.write(hline("New Output Path"))
    f.write(f'!SetOutputPath={{{self.output_path}}}\n')
    # for t in self.times.values():
    #   t.write(f)
    for v in self.solverGs.values():
      v.write(f)
    f.write(hline("Solver Matrices"))
    for v in self.matrices.values():
      v.write(f)
    f.write(hline("Basis Functions"))
    for b in self.bases.values():
      b.write(f)
    f.write(hline("Topologies"))
    for t in self.toplogies.values():
      t.write(f)
    for i in self.interfaces.values():
      i.write(f)
    f.write(hline("Variables"))
    for v in self.vars.values():
      v.write(f)
    for v in self.dataPs.values():
      v.write(f)
    f.write(hline("Export Frequency"))
    for k, v in self.exportfrequencies.items():
      for l in splicegen(60, v):
        f.write(f'!SetExportFrequency={{{"|".join(l)}|{k}}}\n')
    f.write(hline("Problem Definitions"))
    for v in self.problems.values():
      v.write(f)
    f.write(hline("Expression"))
    for v in self.exprs.values():
      v.write(f)
