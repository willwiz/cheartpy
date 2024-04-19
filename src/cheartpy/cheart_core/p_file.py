import dataclasses as dc
from typing import TextIO, overload
from .implementation.time_schemes import TimeScheme
from .implementation.basis import CheartBasis
from .implementation.topologies import (
    _CheartTopology,
    CheartTopology,
    NullTopology,
    hash_tops,
)
from .implementation.topology_interfaces import TopInterface
from .implementation.data_pointers import DataPointer, DataInterp
from .implementation.expressions import Expression
from .implementation.variables import Variable
from .implementation.problems import _Problem
from .implementation.solver_matrices import SolverMatrix
from .implementation.solver_groups import SolverGroup
from .pytools import get_enum, header, hline, splicegen
from .aliases import *


@dc.dataclass(slots=True)
class PFile(object):
    h: str = ""
    output_path: str | None = None
    times: dict[str, TimeScheme] = dc.field(default_factory=dict)
    solverGs: dict[str, SolverGroup] = dc.field(default_factory=dict)
    matrices: dict[str, SolverMatrix] = dc.field(default_factory=dict)
    problems: dict[str, _Problem] = dc.field(default_factory=dict)
    variables: dict[str, Variable] = dc.field(default_factory=dict)
    toplogies: dict[str, _CheartTopology] = dc.field(default_factory=dict)
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
            self.AddTimeScheme(g.time)
            if g.aux_vars:
                self.AddVariable(*g.aux_vars.values())
            for sg in g.SolverSubGroups:
                for p in sg.problems.values():
                    if isinstance(p, SolverMatrix):
                        self.AddMatrix(p)
                    elif isinstance(p, _Problem):
                        self.AddProblem(p)
                if sg.aux_vars:
                    self.AddVariable(*sg.aux_vars.values())

    # Add Time Scheme
    def AddTimeScheme(self, *time: TimeScheme) -> None:
        for t in time:
            self.times[t.name] = t

    # Matrix
    def AddMatrix(self, *mat: SolverMatrix) -> None:
        for v in mat:
            self.matrices[v.name] = v
            if v.problem:
                self.AddProblem(*v.problem.values())
            if v.aux_vars:
                self.AddVariable(*v.aux_vars.values())

    # Problem

    def AddProblem(self, *prob: _Problem) -> None:
        """Internal automatically done through add solver group"""
        for p in prob:
            self.problems[repr(p)] = p
            self.AddVariable(*p.get_variables().values())
            self.AddVariable(*p.get_aux_vars().values())
            for patch in p.get_bc_patches():
                for v in patch.value:
                    if isinstance(v, Variable):
                        self.AddVariable(v)
                    elif isinstance(v, Expression):
                        self.AddExpression(v)

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
            if v.expressions:
                self.AddExpression(*v.expressions.values())

    # Add Topology
    def AddTopology(self, *top: _CheartTopology) -> None:
        for t in top:
            self.toplogies[repr(t)] = t
            if isinstance(t, CheartTopology):
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
        topologies: list[_CheartTopology],
    ) -> None:
        name = hash_tops(topologies)
        self.AddTopology(*topologies)
        self.interfaces[name] = TopInterface(
            name, get_enum(method, TopologyInterfaceType), topologies
        )

    # Add Variables
    @overload
    def SetVariable(
        self,
        name: str,
        task: Literal["INIT_EXPR", "TEMPORAL_UPDATE_EXPR"],
        val: Expression,
    ) -> None: ...

    @overload
    def SetVariable(
        self,
        name: str,
        task: Literal["TEMPORAL_UPDATE_FILE", "TEMPORAL_UPDATE_FILE_LOOP"],
        val: str,
    ) -> None: ...

    def SetVariable(
        self,
        name: str,
        task: Literal[
            "INIT_EXPR",
            "TEMPORAL_UPDATE_EXPR",
            "TEMPORAL_UPDATE_FILE",
            "TEMPORAL_UPDATE_FILE_LOOP",
        ],
        val: str | Expression,
    ):
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
            self.AddSolverGroup(g)

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
