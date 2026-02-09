import dataclasses as dc
from collections.abc import Collection, Generator, Mapping, Sequence
from pathlib import Path
from typing import TextIO

from cheartpy.fe.trait import (
    HasWriter,
    ICheartBasis,
    ICheartTopology,
    IDataInterp,
    IDataPointer,
    IExpression,
    IProblem,
    ISolverGroup,
    ISolverMatrix,
    ISolverSubGroup,
    ITimeScheme,
    ITopInterface,
    IVariable,
)
from cheartpy.fe.utils import Header, header, hline, splicegen


@dc.dataclass(slots=True)
class PFile:
    h: str = ""
    output_dir: Path | None = None
    times: dict[str, ITimeScheme] = dc.field(default_factory=dict[str, ITimeScheme])
    data_pointers: dict[str, IDataPointer] = dc.field(default_factory=dict[str, IDataPointer])
    bases: dict[str, ICheartBasis] = dc.field(default_factory=dict[str, ICheartBasis])
    toplogies: dict[str, ICheartTopology] = dc.field(default_factory=dict[str, ICheartTopology])
    interfaces: dict[str, ITopInterface] = dc.field(default_factory=dict[str, ITopInterface])
    variables: dict[str, IVariable] = dc.field(default_factory=dict[str, IVariable])
    exprs: dict[str, IExpression] = dc.field(default_factory=dict[str, IExpression])
    problems: dict[str, IProblem] = dc.field(default_factory=dict[str, IProblem])
    matrices: dict[str, ISolverMatrix] = dc.field(default_factory=dict[str, ISolverMatrix])
    solver_groups: dict[str, ISolverGroup] = dc.field(default_factory=dict[str, ISolverGroup])

    def set_outputpath(self, path: Path | str) -> None:
        self.output_dir = Path(path)

    # Add Time Scheme
    def add_timescheme(self, *time: ITimeScheme) -> None:
        for t in time:
            if str(t) not in self.times:
                self.times[str(t)] = t

    def add_datapointer(self, *var: IDataPointer) -> None:
        for v in var:
            if str(v) not in self.data_pointers:
                self.data_pointers[str(v)] = v

    def add_basis(self, *basis: ICheartBasis | None) -> None:
        for b in basis:
            if b is not None and str(b) not in self.bases:
                self.bases[str(b)] = b

    def add_topology(self, *top: ICheartTopology) -> None:
        for t in top:
            self.add_basis(t.get_basis())
            if str(t) not in self.toplogies:
                self.toplogies[str(t)] = t

    def add_interface(self, *interfaces: ITopInterface) -> None:
        for item in interfaces:
            self.add_topology(*item.get_tops())
            if str(item) not in self.interfaces:
                self.interfaces[str(item)] = item

    def add_variable(self, *var: IVariable) -> None:
        for v in var:
            self.add_topology(v.get_top())
            self.add_expression(*v.get_expr_deps())
            if str(v) not in self.variables:
                self.variables[str(v)] = v

    def add_expression(self, *expr: IExpression) -> None:
        for v in expr:
            for x in v.get_values():
                if isinstance(x, IDataInterp):
                    self.add_datapointer(x.get_datapointer())
                elif isinstance(x, IVariable):
                    self.add_variable(x)
                elif isinstance(x, IExpression):
                    self.add_expression(x)
                elif isinstance(x, tuple):
                    if isinstance(x[0], IDataInterp):
                        self.add_datapointer(x[0].get_datapointer())
                    elif isinstance(x[0], IExpression):
                        self.add_expression(x[0])
                    else:
                        self.add_variable(x[0])
            self.add_variable(*v.get_var_deps())
            self.add_expression(*v.get_expr_deps())
            if str(v) not in self.exprs:
                self.exprs[str(v)] = v

    def add_problem(self, *prob: IProblem) -> None:
        for p in prob:
            self.add_expression(*p.get_expr_deps())
            self.add_variable(*p.get_var_deps())
            for patch in p.get_bc_patches():
                self.add_variable(*patch.get_var_deps())
                self.add_expression(*patch.get_expr_deps())
            if str(p) not in self.problems:
                self.problems[str(p)] = p

    # Matrix
    def add_matrix(self, *mat: ISolverMatrix) -> None:
        for m in mat:
            self.add_problem(*m.get_problems())
            if str(m) not in self.matrices:
                self.matrices[str(m)] = m

    def add_solversubgroup(self, *subgroup: ISolverSubGroup) -> None:
        # PFile does not need this, SolverGroup will handle printing
        for sg in subgroup:
            self.add_problem(*sg.get_problems())
            self.add_matrix(*sg.get_matrices())

    # SolverGroup
    def add_solvergroup(self, *grp: ISolverGroup) -> None:
        for g in grp:
            self.add_timescheme(g.get_time_scheme())
            self.add_solversubgroup(*g.get_subgroups())
            if str(g) not in self.solver_groups:
                self.solver_groups[str(g)] = g

    # Set Export Frequency
    def set_exportfrequency(self, *var: IVariable, freq: int = 1) -> None:
        for v in var:
            v.set_export_frequency(freq)

    # ----------------------------------------------------------------------------
    # Resolve Pfile
    def resolve(self) -> None:
        for g in self.solver_groups.values():
            self.add_solvergroup(g)

    # ----------------------------------------------------------------------------
    # Producing the Pfile

    def write(self, f: TextIO) -> None:
        self.resolve()
        _pfile_writer(self, f)


def _get_ordered_topinterfaces(interfaces: Mapping[str, ITopInterface]) -> list[ITopInterface]:
    return [
        *[v for v in interfaces.values() if v.method == "OneToOne"],
        *[v for v in interfaces.values() if v.method == "ManyToOne"],
    ]


def _get_variable_frequency_list(
    variables: Mapping[str, IVariable],
) -> Mapping[int, Collection[str]]:
    exportfrequencies: dict[int, set[str]] = {}
    for v in variables.values():
        if v.get_export_frequency() in exportfrequencies:
            exportfrequencies[v.get_export_frequency()].update({str(v)})
        else:
            exportfrequencies[v.get_export_frequency()] = {str(v)}
    return exportfrequencies


def _get_writer(
    header: str, *obj: Sequence[HasWriter] | Mapping[str, HasWriter]
) -> Generator[HasWriter]:
    yield Header(header)
    for o in obj:
        match o:
            case Mapping():
                for v in o.values():
                    yield v
            case Sequence():
                for v in o:
                    yield v


def _pfile_writer(pfile: PFile, f: TextIO) -> None:
    f.write(header(pfile.h))
    f.write(hline("New Output Path"))
    if pfile.output_dir is not None:
        f.write(f"!SetOutputPath={{{pfile.output_dir}}}\n")
    for v in _get_writer("Solver Groups", pfile.times, pfile.solver_groups):
        v.write(f)
    for m in _get_writer("Solver Matrices", pfile.matrices):
        m.write(f)
    for b in _get_writer("Basis Functions", pfile.bases):
        b.write(f)
    for t in _get_writer(
        "Topologies", pfile.toplogies, _get_ordered_topinterfaces(pfile.interfaces)
    ):
        t.write(f)
    for v in _get_writer("Variables", pfile.variables, pfile.data_pointers):
        v.write(f)
    f.write(hline("Export Frequency"))
    for freq, v in _get_variable_frequency_list(pfile.variables).items():
        f.writelines(
            f"!SetExportFrequency={{{'|'.join(s)}|{freq}}}\n" for s in splicegen(60, sorted(v))
        )
    for v in _get_writer("Problem Definitions", pfile.problems):
        v.write(f)
    for v in _get_writer("Expression", pfile.exprs):
        v.write(f)
