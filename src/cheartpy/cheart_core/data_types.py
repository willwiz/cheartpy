#!/usr/bin/env python3
from dataclasses import dataclass, field
from typing import TextIO, Type, overload
from .aliases import *
from .keywords import *
from .pytools import *
from .base_types import *
from .dictionary import *
from .problems import *

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


# Solver Matrix
@dataclass
class SolverMatrix:
    name: str
    method: str
    problem: dict[str, Problem] = field(default_factory=dict)
    settings: dict[str, Setting] = field(default_factory=dict)
    aux_vars: dict[str, Variable] = field(default_factory=dict)

    def __post_init__(self):
        for _, p in self.problem.items():
            for k, x in p.aux_vars.items():
                self.aux_vars[k] = x

    def AddSetting(self, opt, *val):
        self.settings[opt] = Setting(opt, [*val])

    def write(self, f: TextIO):
        f.write(
            f'!DefSolverMatrix={{{self.name}|{self.method}|{
                "|".join([p.name for p in self.problem.values()])}}}\n'
        )
        for k, v in self.settings.items():
            f.write(f" !SetSolverMatrix={{{self.name}|{v.string()}}}\n")


# Define Solver SubGroup
@dataclass
class SolverSubGroup:
    name: str = field(default="none")
    method: Literal["seq_fp_linesearch", "SOLVER_SEQUENTIAL"] = field(
        default="seq_fp_linesearch"
    )
    problems: dict[str, SolverMatrix | Problem] = field(
        default_factory=dict
    )
    aux_vars: dict[str, Variable] = field(default_factory=dict)
    scale_file_residual: bool = False

    def __post_init__(self):
        for p in self.problems.values():
            for k, x in p.aux_vars.items():
                self.aux_vars[k] = x


@dataclass
class SolverGroup(object):
    name: str
    time: TimeScheme
    export_initial_condition: bool = False
    SolverSubGroups: list[SolverSubGroup] = field(default_factory=list)
    aux_vars: dict[str, Variable] = field(default_factory=dict)
    settings: dict[str, Setting] = field(default_factory=dict)
    use_dynamic_topologies: bool | float = False

    # TOL
    def __post_init__(self):
        for sg in self.SolverSubGroups:
            for k, v in sg.aux_vars.items():
                self.aux_vars[k] = v

    @overload
    def AddSetting(
        self,
        task: Literal[
            "L2TOL",
            "L2PERCENT",
            "INFRES",
            "INFUPDATE",
            "INFDEL",
            "INFRELUPDATE",
            "L2RESRELPERCENT",
        ]
        | Literal["ITERATION", "SUBITERATION", "LINESEARCHITER", "SUBITERFRACTION"],
        val: Union[Expression, Variable, float, str],
    ) -> None:
        ...

    @overload
    def AddSetting(self, task: TolSettings, val: float | str) -> None:
        ...

    @overload
    def AddSetting(self, task: IterationSettings, val: int | str) -> None:
        ...

    def AddSetting(self, task, val):
        self.settings[task] = Setting(task, [val])

    # VAR
    def set_convergence(
        self,
        task: TolSettings
        | Literal[
            "L2TOL",
            "L2PERCENT",
            "INFRES",
            "INFUPDATE",
            "INFDEL",
            "INFRELUPDATE",
            "L2RESRELPERCENT",
        ],
        val: Union[float, str],
    ) -> None:
        self.settings[task] = Setting(task, [val])

    def set_iteration(
        self,
        task: IterationSettings
        | Literal[
            "ITERATION",
            "SUBITERATION",
            "LINESEARCHITER",
            "SUBITERFRACTION",
            "GroupIterations",
        ],
        val: Union[int, str],
    ) -> None:
        self.settings[task] = Setting(task, [val])

    def catch_solver_errors(
        self, err: Literal["nan_maxval"], act: Literal["evaluate_full"]
    ) -> None:
        self.settings["CatchSolverErrors"] = Setting(
            "CatchSolverErrors", [err, act])

    def AddVariable(self, *var: Variable):
        for v in var:
            if isinstance(v, str):
                self.aux_vars[v] = v
            else:
                self.aux_vars[v.name] = v

    def RemoveVariable(self, *var: Union[str, Variable]):
        for v in var:
            if isinstance(v, str):
                self.aux_vars.pop(v)
            else:
                self.aux_vars.pop(v.name)

    # SG
    def AddSolverSubGroup(self, *sg: SolverSubGroup) -> None:
        for v in sg:
            self.SolverSubGroups.append(v)
            for x in v.aux_vars.values():
                self.AddVariable(x)

    def RemoveSolverSubGroup(self, *sg: SolverSubGroup) -> None:
        for v in sg:
            self.SolverSubGroups.remove(v)

    def MakeSolverSubGroup(
        self,
        method: Literal["seq_fp_linesearch", "SOLVER_SEQUENTIAL"],
        *problems: Union[SolverMatrix, Problem],
    ) -> None:
        self.SolverSubGroups.append(
            SolverSubGroup(method=method,
                           problems={p.name: p for p in problems})
        )

    # WRITE
    def write(self, f: TextIO) -> None:
        # if isinstance(self.time,TimeScheme):
        #   self.time.write(f)
        f.write(hline("Solver Groups"))
        f.write(f"!DefSolverGroup={{{self.name}|{VoS(self.time)}}}\n")
        # Handle Additional Vars
        vars = [VoS(v) for v in self.aux_vars.values()]
        for l in splicegen(45, vars):
            if l:
                f.write(
                    f'  !SetSolverGroup={{{
                        self.name}|AddVariables|{"|".join(l)}}}\n'
                )
        # Print export init setting
        if self.export_initial_condition:
            f.write(
                f"  !SetSolverGroup={{{self.name}|export_initial_condition}}\n")
        # Print Conv Settings
        for _, s in self.settings.items():
            f.write(f"  !SetSolverGroup={{{self.name}|{s.string()}}}\n")
        if self.use_dynamic_topologies:
            f.write(
                f"  !SetSolverGroup={{{self.name}|UsingDynamicTopologies}}\n")
        for g in self.SolverSubGroups:
            pobs = [VoS(p) for p in g.problems]
            if g.scale_file_residual:
                f.write(
                    f'!DefSolverSubGroup={{{self.name}|{g.method}|{
                        "|".join(pobs)}|ScaleFirstResidual[{g.scale_file_residual}]}}\n'
                )
            else:
                f.write(
                    f'!DefSolverSubGroup={{{self.name}|{
                        g.method}|{"|".join(pobs)}}}\n'
                )


def hash_tops(tops: list[Topology] | list[str]) -> str:
    names = [VoS(t) for t in tops]
    return "_".join(names)


@dataclass(slots=True)
class PFile(object):
    h: str = ""
    output_path: str | None = None
    times: dict[str, TimeScheme] = field(default_factory=dict)
    solverGs: dict[str, SolverGroup] = field(default_factory=dict)
    matrices: dict[str, SolverMatrix] = field(default_factory=dict)
    problems: dict[str, Problem] = field(default_factory=dict)
    variables: dict[str, Variable] = field(default_factory=dict)
    toplogies: dict[str, Topology] = field(default_factory=dict)
    bases: dict[str, CheartBasis] = field(default_factory=dict)
    dataPs: dict[str, DataPointer] = field(default_factory=dict)
    exprs: dict[str, Expression] = field(default_factory=dict)
    interfaces: dict[str, TopInterface] = field(default_factory=dict)

    # exportfrequencies: dict[str, Set[Variable]] = field(default_factory=dict)
    def SetOutputPath(self, path):
        self.output_path = path

    # SolverGroup
    def AddSolverGroup(self, *grp: SolverGroup) -> None:
        for g in grp:
            self.solverGs[g.name] = g

    # Add Time Scheme
    def AddTimeScheme(self, *time: TimeScheme) -> None:
        for t in time:
            self.times[t.name] = t

    # Matrix
    def AddMatrix(self, *mat: SolverMatrix) -> None:
        for v in mat:
            self.matrices[v.name] = v
            for p in v.problem.values():
                self.AddProblem(p)

    # Problem
    def AddProblem(self, *prob: Problem) -> None:
        """Internal automatically done through add solver group"""
        for p in prob:
            self.problems[p.name] = p
            for v in p.vars.values():
                self.AddVariable(v)
            if p.BC.patches is not None:
                for patch in p.BC.patches:
                    if isinstance(patch.value, Variable):
                        self.AddVariable(patch.value)
                    elif isinstance(patch.value, Expression):
                        self.AddExpression(patch.value)

    def AddVariable(self, *var: Variable) -> None:
        for v in var:
            self.variables[v.name] = v
            # self.SetExportFrequency(v, freq=v.freq)
            self.AddTopology(v.topology)
            for val in v.setting.values():
                if isinstance(val, Variable):
                    self.AddVariable(val)
                elif isinstance(val, Expression):
                    self.AddExpression(val)

    # Add Topology
    def AddTopology(self, *top: Topology) -> None:
        for t in top:
            self.toplogies[t.name] = t
            self.AddBasis(t.basis)

    # Add Basis
    def AddBasis(self, *basis: CheartBasis | None) -> None:
        for b in basis:
            if b is not None:
                self.bases[b.name] = b

    # Expression
    def AddExpression(self, *expr: Expression) -> None:
        for v in expr:
            self.exprs[v.name] = v
            for x in v.value:
                if isinstance(x, DataInterp):
                    self.AddDataPointer(x.var)

    def SetTopology(self, name, task, val) -> None:
        self.toplogies[name].AddSetting(task, val)

    # Add Interfaces
    # def AddInterface(self, *interface:TopInterface) -> None:
    #   for v in interface:
    #     self.interfaces[v.name] = v
    def AddInterface(
        self,
        method: Literal["OneToOne", "ManyToOne"],
        topologies: list[Topology],
    ) -> None:
        name = hash_tops(topologies)
        self.interfaces[name] = TopInterface(name, method, topologies)

    # Add Variables
    @overload
    def SetVariable(
        self, name: str, task: Literal["INIT_EXPR", "TEMPORAL_UPDATE_EXPR"], val: Expression) -> None: ...

    @overload
    def SetVariable(
        self, name: str, task: Literal["TEMPORAL_UPDATE_FILE", "TEMPORAL_UPDATE_FILE_LOOP"], val: str) -> None: ...

    def SetVariable(self, name: str, task: Literal["INIT_EXPR", "TEMPORAL_UPDATE_EXPR", "TEMPORAL_UPDATE_FILE", "TEMPORAL_UPDATE_FILE_LOOP"], val: str | Expression):
        self.variables[name].AddSetting(task, val)

    # Add Data Pointers
    def AddDataPointer(self, *var: DataPointer) -> None:
        for v in var:
            self.dataPs[v.name] = v

    # Set Export Frequency
    def SetExportFrequency(self, *vars: Variable, freq: int = 1):
        for v in vars:
            v.freq = freq
            # else:
            #   print(f">>>WARNING: setting frequency but {v.name} is not being used.",file=sys.stderr)

    def get_variable_frequency_list(self):
        # pprint.pprint(self.vars)
        exportfrequencies = dict()
        for v in self.variables.values():
            if str(v.freq) in exportfrequencies:
                exportfrequencies[str(v.freq)].update({VoS(v)})
            else:
                exportfrequencies[str(v.freq)] = {VoS(v)}
        return exportfrequencies

    # ----------------------------------------------------------------------------
    # Resolve Pfile
    def resolve(self):
        for g in self.solverGs.values():
            self.AddTimeScheme(g.time)
            for sg in g.SolverSubGroups:
                for p in sg.problems:
                    if isinstance(p, SolverMatrix):
                        self.AddMatrix(p)
                    elif isinstance(p, Problem):
                        self.AddProblem(p)
            for v in g.aux_vars.values():
                self.AddVariable(v)

    # ----------------------------------------------------------------------------
    # Producing the Pfile
    def write(self, f: TextIO):
        self.resolve()
        f.write(header(self.h))
        f.write(hline("New Output Path"))
        f.write(f"!SetOutputPath={{{self.output_path}}}\n")
        for t in self.times.values():
            t.write(f)
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
        for v in self.variables.values():
            v.write(f)
        for v in self.dataPs.values():
            v.write(f)
        f.write(hline("Export Frequency"))
        exportfrequencies = self.get_variable_frequency_list()
        for k, v in exportfrequencies.items():
            for l in splicegen(60, v):
                f.write(f'!SetExportFrequency={{{"|".join(l)}|{k}}}\n')
        f.write(hline("Problem Definitions"))
        for v in self.problems.values():
            v.write(f)
        f.write(hline("Expression"))
        for v in self.exprs.values():
            v.write(f)
