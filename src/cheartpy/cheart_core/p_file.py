import dataclasses as dc
from typing import TextIO, overload
from cheartpy.cheart_core.time_schemes import TimeScheme
from cheartpy.cheart_core.bases import CheartBasis
from cheartpy.cheart_core.topologies import CheartTopology, hash_tops
from cheartpy.cheart_core.topology_interfaces import TopInterface
from cheartpy.cheart_core.data_pointers import DataPointer, DataInterp
from cheartpy.cheart_core.expressions import Expression
from cheartpy.cheart_core.variables import Variable
from cheartpy.cheart_core.problems import Problem
from cheartpy.cheart_core.solver_matrices import SolverMatrix
from cheartpy.cheart_core.solver_groups import SolverGroup
from cheartpy.cheart_core.pytools import get_enum, header, hline, cline, splicegen
from .aliases import *


@dc.dataclass(slots=True)
class PFile(object):
    h: str = ""
    output_path: str | None = None
    times: dict[str, TimeScheme] = dc.field(default_factory=dict)
    solverGs: dict[str, SolverGroup] = dc.field(default_factory=dict)
    matrices: dict[str, SolverMatrix] = dc.field(default_factory=dict)
    problems: dict[str, Problem] = dc.field(default_factory=dict)
    variables: dict[str, Variable] = dc.field(default_factory=dict)
    toplogies: dict[str, CheartTopology] = dc.field(default_factory=dict)
    bases: dict[str, CheartBasis] = dc.field(default_factory=dict)
    dataPs: dict[str, DataPointer] = dc.field(default_factory=dict)
    exprs: dict[str, Expression] = dc.field(default_factory=dict)
    interfaces: dict[str, TopInterface] = dc.field(default_factory=dict)

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
            self.AddVariable(*p.variables.values())
            if p.bc.patches is not None:
                for patch in p.bc.patches:
                    if isinstance(patch.value, Variable):
                        self.AddVariable(patch.value)
                    elif isinstance(patch.value, Expression):
                        self.AddExpression(patch.value)

    def AddVariable(self, *var: Variable) -> None:
        for v in var:
            if v.name not in self.variables:
                self.variables[v.name] = v
            # self.SetExportFrequency(v, freq=v.freq)
            if v.topology:
                self.AddTopology(v.topology)
            if v.setting:
                if isinstance(v.setting[1], Expression):
                    self.AddExpression(v.setting[1])

    # Add Topology
    def AddTopology(self, *top: CheartTopology) -> None:
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
            if v.name not in self.exprs:
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
        method: Literal["OneToOne", "ManyToOne"] | TopologyInterfaceType,
        topologies: list[CheartTopology],
    ) -> None:
        name = hash_tops(topologies)
        self.AddTopology(*topologies)
        self.interfaces[name] = TopInterface(
            name, get_enum(method, TopologyInterfaceType), topologies)

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
                exportfrequencies[str(v.freq)].update({str(v)})
            else:
                exportfrequencies[str(v.freq)] = {str(v)}
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
