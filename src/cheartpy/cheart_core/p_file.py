import dataclasses as dc
from typing import TextIO, overload
from .aliases import *
from .interface import *
from .pytools import header, hline, splicegen


@dc.dataclass(slots=True)
class PFile:
    h: str = ""
    output_path: str | None = None
    times: dict[str, _TimeScheme] = dc.field(default_factory=dict)
    solverGs: dict[str, _SolverGroup] = dc.field(default_factory=dict)
    matrices: dict[str, _SolverMatrix] = dc.field(default_factory=dict)
    problems: dict[str, _Problem] = dc.field(default_factory=dict)
    variables: dict[str, _Variable] = dc.field(default_factory=dict)
    toplogies: dict[str, _CheartTopology] = dc.field(default_factory=dict)
    bases: dict[str, _CheartBasis] = dc.field(default_factory=dict)
    dataPs: dict[str, _DataPointer] = dc.field(default_factory=dict)
    exprs: dict[str, _Expression] = dc.field(default_factory=dict)
    interfaces: dict[str, _TopInterface] = dc.field(default_factory=dict)

    # exportfrequencies: dict[str, Set[Variable]] = field(default_factory=dict)
    def SetOutputPath(self, path):
        self.output_path = path

    # SolverGroup
    def AddSolverGroup(self, *grp: _SolverGroup) -> None:
        for g in grp:
            self.solverGs[str(g)] = g
            self.AddTimeScheme(g.get_time_scheme())
            self.AddVariable(*g.get_aux_vars())
            for sg in g.get_subgroups():
                for p in sg.get_problems():
                    if isinstance(p, _SolverMatrix):
                        self.AddMatrix(p)
                    elif isinstance(p, _Problem):
                        self.AddProblem(p)
                if sg.get_aux_vars():
                    self.AddVariable(*sg.get_aux_vars().values())

    # Add Time Scheme
    def AddTimeScheme(self, *time: _TimeScheme) -> None:
        for t in time:
            self.times[str(t)] = t

    # Matrix
    def AddMatrix(self, *mat: _SolverMatrix) -> None:
        for v in mat:
            self.matrices[str(v)] = v
            self.AddProblem(*v.get_problems())
            self.AddVariable(*v.get_aux_vars())

    # Problem

    def AddProblem(self, *prob: _Problem) -> None:
        """Internal automatically done through add solver group"""
        for p in prob:
            self.problems[str(p)] = p
            self.AddVariable(*p.get_variables().values())
            self.AddVariable(*p.get_aux_vars())
            self.AddExpression(*p.get_aux_expr().values())
            for patch in p.get_bc_patches():
                for v in patch.get_values():
                    if isinstance(v, _Variable):
                        self.AddVariable(v)
                    elif isinstance(v, _Expression):
                        self.AddExpression(v)

    def AddVariable(self, *var: _Variable) -> None:
        for v in var:
            if str(v) not in self.variables:
                self.variables[str(v)] = v
            # self.SetExportFrequency(v, freq=v.freq)
            self.AddTopology(v.get_top())
            self.AddExpression(*v.get_expressions())

    # Add Topology
    def AddTopology(self, *top: _CheartTopology) -> None:
        for t in top:
            if isinstance(t, _CheartTopology):
                self.toplogies[str(t)] = t
                self.AddBasis(t.get_basis())

    # Add Basis
    def AddBasis(self, *basis: _CheartBasis | None) -> None:
        for b in basis:
            if b is not None:
                self.bases[str(b)] = b

    # Expression
    def AddExpression(self, *expr: _Expression) -> None:
        for v in expr:
            if str(v) not in self.exprs:
                self.exprs[str(v)] = v
            for x in v.get_values():
                if isinstance(x, _DataInterp):
                    self.AddDataPointer(x.get_val())
                elif isinstance(x, _Expression):
                    self.AddExpression(x)
                elif isinstance(x, _Variable):
                    self.AddVariable(x)
                elif isinstance(x, tuple):
                    if isinstance(x[0], _DataInterp):
                        self.AddDataPointer(x[0].get_val())
                    elif isinstance(x[0], _Expression):
                        self.AddExpression(x[0])
                    elif isinstance(x[0], _Variable):
                        self.AddVariable(x[0])
            self.AddVariable(*v.get_var_deps())
            self.AddExpression(*v.get_expr_deps())

    def SetTopology(self, name, task, val) -> None:
        self.toplogies[name].AddSetting(task, val)

    # Add Interfaces
    # def AddInterface(self, *interface:TopInterface) -> None:
    #   for v in interface:
    #     self.interfaces[v.name] = v
    # def AddInterface(
    #     self,
    #     method: Literal["OneToOne", "ManyToOne"] | TopologyInterfaceType,
    #     topologies: list[_CheartTopology],
    #     master_topology: _CheartTopology | None = None,
    #     interface_file: str | None = None,
    #     nest_in_boundary: int | None = None,
    # ) -> None:
    #     match method:
    #         case "OneToOne":
    #             name = hash_tops(topologies)
    #             self.AddTopology(*topologies)
    #             self.interfaces[name] = OneToOneTopInterface(name, topologies)
    #         case "ManyToOne":
    #             if master_topology is None:
    #                 raise ValueError("ManyToOne requires a master_topology")
    #             if interface_file is None:
    #                 raise ValueError("ManyToOne requires a interface_file")
    #             name = hash_tops(topologies)
    #             self.AddTopology(*topologies)
    #             self.AddTopology(master_topology)
    #             self.interfaces[name] = ManyToOneTopInterface(
    #                 name, topologies, master_topology, interface_file, nest_in_boundary
    #             )

    def AddInterface(self, *interfaces: _TopInterface) -> None:
        for item in interfaces:
            self.interfaces[str(item)] = item
            for t in item.get_tops():
                self.AddTopology(t)

    # Add Variables
    @overload
    def SetVariable(
        self,
        name: str,
        task: Literal["INIT_EXPR", "TEMPORAL_UPDATE_EXPR"],
        val: _Expression,
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
        val: str | _Expression,
    ):
        self.variables[name].AddSetting(task, val)

    # Add Data Pointers
    def AddDataPointer(self, *var: _DataPointer) -> None:
        for v in var:
            self.dataPs[str(v)] = v

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
