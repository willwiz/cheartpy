import dataclasses as dc
from typing import TextIO
from .aliases import *
from .interface import *
from .pytools import header, hline, splicegen


@dc.dataclass(slots=True)
class PFile:
    h: str = ""
    output_path: str | None = None
    times: dict[str, _TimeScheme] = dc.field(default_factory=dict)
    dataPs: dict[str, _DataPointer] = dc.field(default_factory=dict)
    bases: dict[str, _CheartBasis] = dc.field(default_factory=dict)
    toplogies: dict[str, _CheartTopology] = dc.field(default_factory=dict)
    interfaces: dict[str, _TopInterface] = dc.field(default_factory=dict)
    variables: dict[str, _Variable] = dc.field(default_factory=dict)
    exprs: dict[str, _Expression] = dc.field(default_factory=dict)
    problems: dict[str, _Problem] = dc.field(default_factory=dict)
    matrices: dict[str, _SolverMatrix] = dc.field(default_factory=dict)
    solverGs: dict[str, _SolverGroup] = dc.field(default_factory=dict)

    def SetOutputPath(self, path):
        self.output_path = path

    # Add Time Scheme
    def AddTimeScheme(self, *time: _TimeScheme) -> None:
        for t in time:
            if str(t) not in self.times:
                self.times[str(t)] = t

    # Add Data Pointers
    def AddDataPointer(self, *var: _DataPointer) -> None:
        for v in var:
            if str(v) not in self.dataPs:
                self.dataPs[str(v)] = v

    # Add Basis
    def AddBasis(self, *basis: _CheartBasis | None) -> None:
        for b in basis:
            if b is not None:
                if str(b) not in self.bases:
                    self.bases[str(b)] = b

    # Add Topology
    def AddTopology(self, *top: _CheartTopology) -> None:
        for t in top:
            self.AddBasis(t.get_basis())
            if str(t) not in self.toplogies:
                self.toplogies[str(t)] = t

    def AddInterface(self, *interfaces: _TopInterface) -> None:
        for item in interfaces:
            self.AddTopology(*item.get_tops())
            if str(item) not in self.interfaces:
                self.interfaces[str(item)] = item

    def AddVariable(self, *var: _Variable) -> None:
        for v in var:
            self.AddTopology(v.get_top())
            self.AddExpression(*v.get_expr_deps())
            if str(v) not in self.variables:
                self.variables[str(v)] = v

    # Expression
    def AddExpression(self, *expr: _Expression) -> None:
        for v in expr:
            for x in v.get_values():
                if isinstance(x, _DataInterp):
                    self.AddDataPointer(x.get_datapointer())
                elif isinstance(x, _Variable):
                    self.AddVariable(x)
                elif isinstance(x, _Expression):
                    self.AddExpression(x)
                elif isinstance(x, tuple):
                    if isinstance(x[0], _DataInterp):
                        self.AddDataPointer(x[0].get_datapointer())
                    elif isinstance(x[0], _Expression):
                        self.AddExpression(x[0])
                    elif isinstance(x[0], _Variable):
                        self.AddVariable(x[0])
            self.AddVariable(*v.get_var_deps())
            self.AddExpression(*v.get_expr_deps())
            if str(v) not in self.exprs:
                self.exprs[str(v)] = v

    def AddProblem(self, *prob: _Problem) -> None:
        """Internal automatically done through add solver group"""
        for p in prob:
            self.AddVariable(*p.get_var_deps())
            self.AddExpression(*p.get_expr_deps())
            # self.AddVariable(*p.get_variables().values())
            for patch in p.get_bc_patches():
                self.AddVariable(*patch.get_var_deps())
                self.AddExpression(*patch.get_expr_deps())
            if str(p) not in self.problems:
                self.problems[str(p)] = p

    # Matrix
    def AddMatrix(self, *mat: _SolverMatrix) -> None:
        for m in mat:
            self.AddProblem(*m.get_problems())
            if str(m) not in self.matrices:
                self.matrices[str(m)] = m

    def AddSolverSubGroup(self, *subgroup: _SolverSubGroup) -> None:
        # PFile does not need this, SolverGroup will handle printing
        for sg in subgroup:
            self.AddProblem(*sg.get_problems())
            self.AddMatrix(*sg.get_matrices())

    # SolverGroup
    def AddSolverGroup(self, *grp: _SolverGroup) -> None:
        for g in grp:
            self.AddTimeScheme(g.get_time_scheme())
            self.AddSolverSubGroup(*g.get_subgroups())
            # self.AddVariable(*g.get_aux_vars())
            # for sg in g.get_subgroups():
            # for p in sg.get_problems():
            #     if isinstance(p, _SolverMatrix):
            #         self.AddMatrix(p)
            #     elif isinstance(p, _Problem):
            #         self.AddProblem(p)
            # if sg.get_aux_vars():
            #     self.AddVariable(*sg.get_aux_vars().values())
            if str(g) not in self.solverGs:
                self.solverGs[str(g)] = g

    # Problem

    # def SetTopology(self, name, task, val) -> None:
    #     self.toplogies[name].AddSetting(task, val)

    # Add Variables
    # @overload
    # def SetVariable(
    #     self,
    #     name: str,
    #     task: Literal["INIT_EXPR", "TEMPORAL_UPDATE_EXPR"],
    #     val: _Expression,
    # ) -> None: ...

    # @overload
    # def SetVariable(
    #     self,
    #     name: str,
    #     task: Literal["TEMPORAL_UPDATE_FILE", "TEMPORAL_UPDATE_FILE_LOOP"],
    #     val: str,
    # ) -> None: ...

    # def SetVariable(
    #     self,
    #     name: str,
    #     task: Literal[
    #         "INIT_EXPR",
    #         "TEMPORAL_UPDATE_EXPR",
    #         "TEMPORAL_UPDATE_FILE",
    #         "TEMPORAL_UPDATE_FILE_LOOP",
    #     ],
    #     val: str | _Expression,
    # ):
    #     self.variables[name].AddSetting(task, val)

    # Set Export Frequency
    def SetExportFrequency(self, *vars: _Variable, freq: int = 1):
        for v in vars:
            v.set_export_frequency(freq)

    def get_variable_frequency_list(self):
        # pprint.pprint(self.vars)
        exportfrequencies = dict()
        for v in self.variables.values():
            if v.get_export_frequency() in exportfrequencies:
                exportfrequencies[v.get_export_frequency()].update({str(v)})
            else:
                exportfrequencies[v.get_export_frequency()] = {str(v)}
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
        if self.output_path is not None:
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
            for l in splicegen(60, sorted(v)):
                f.write(f'!SetExportFrequency={{{"|".join(l)}|{k}}}\n')
        f.write(hline("Problem Definitions"))
        for v in self.problems.values():
            v.write(f)
        f.write(hline("Expression"))
        for v in self.exprs.values():
            v.write(f)
