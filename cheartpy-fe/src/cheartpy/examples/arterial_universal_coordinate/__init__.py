import dataclasses as dc
from typing import TYPE_CHECKING, Literal, Required, Unpack

from cheartpy.fe.api import (
    CompiledTopologies,
    create_bcpatch,
    create_pfile,
    create_solver_group,
    create_solver_matrix,
    create_solver_subgroup,
    create_time_scheme,
    create_topologies,
    create_variable,
)
from cheartpy.fe.physics.api import create_l2varprojection_problem, create_laplace_problem
from pytools.result import Err, Ok, Result
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from cheartpy.fe.aliases import TopologyDef
    from cheartpy.fe.trait import IBCPatch, ICheartTopology, IPFile, IVariable

TX = Literal["X"]


class VariableNameKwargs(TypedDict, total=False):
    a: str
    v: str


class BoundaryValueKwargs(TypedDict, total=True):
    z: Mapping[int, float]
    r: Mapping[int, float]


class APIKwargs[T](TypedDict, total=False):
    top: Required[TopologyDef[T]]
    bc: Required[Mapping[int, float]]
    prefix: VariableNameKwargs
    output_dir: Path


@dc.dataclass(slots=True)
class VariableList:
    x: IVariable
    a: IVariable
    v: IVariable



def create_problem_topology[T](**kwargs: Unpack[APIKwargs[T]]) -> CompiledTopologies[TX]:
    match kwargs["top"]:
        case {"mesh": mesh, "elem": elem, "order": order}:
            defn: Mapping[TX, TopologyDef[TX]] = {"X": {"mesh": mesh, "elem": elem, "order": order}}
        case _:
            msg = f"Invalid topology definition: {kwargs['top']!r}."
            raise ValueError(msg)
    return create_topologies(defn)


def create_variable_list[T](
    t: ICheartTopology, **kwargs: Unpack[APIKwargs[T]]
) -> Result[VariableList]:
    if t.mesh is None:
        msg = f"Topology {t!s} does not have a mesh file specified."
        return Err(ValueError(msg))
    space = create_variable("X", t, 3, data=t.mesh, freq=-1)
    prefix = kwargs.get("prefix") or {}
    vs = VariableList(
        x=space,
        a=create_variable(prefix.get("a", "Field"), t, 1),
        v=create_variable(prefix.get("v", "Vector"), t, space.get_dim()),
    )
    return Ok(vs)


def create_bc_patches[T](vlist: VariableList, **kwargs: Unpack[APIKwargs[T]]) -> list[IBCPatch]:
    return [create_bcpatch(i, vlist.a, "dirichlet", v) for i, v in kwargs["bc"].items()]


def uac_pfile[T](**kwargs: Unpack[APIKwargs[T]]) -> IPFile:
    """Create a P-File for generating fiber fields.

    Parameters
    ----------
    top : TopologyDef[T]
        The topology definition for the problem.
    bc : Mapping[int, float]
        The boundary conditions for the problem.
    prefix : VariableNameKwargs, optional
        The prefixes for the variable names. Defaults to {"a": "Field", "v": "Vector"}.
    output_dir : Path, optional
        The output directory for the problem. Defaults to the current working directory.

    Returns
    -------
    IPFile
        The generated P-File for the problem.

    """
    time = create_time_scheme("time", 1, 1, 1)
    top, iface = create_problem_topology(**kwargs)
    var = create_variable_list(top["X"], **kwargs).unwrap()
    bcs = create_bc_patches(var, **kwargs)
    coord_prob = create_laplace_problem(f"Problem{var.a!s}", var.x, var.a)
    coord_prob.bc.add_patch(*bcs)
    vec_probs = create_l2varprojection_problem(f"Problem{var.v!s}", var.x, var.a, var.v, "gradient")
    coord_matrices = create_solver_matrix(f"Matrix{coord_prob!s}", "SOLVER_MUMPS", coord_prob)
    vec_matrices = create_solver_matrix(f"Matrix{vec_probs!s}", "SOLVER_MUMPS", vec_probs)

    solver_subgroups = [
        create_solver_subgroup("seq_fp_linesearch", coord_matrices),
        create_solver_subgroup("SOLVER_SEQUENTIAL", vec_matrices),
    ]
    sg = create_solver_group("Main", time, *solver_subgroups)
    sg.export_initial_condition = False
    pfile = create_pfile()
    pfile.add_solvergroup(sg)
    if output_path := kwargs.get("output_dir"):
        pfile.set_outputpath(output_path)
    pfile.add_interface(*iface)
    return pfile
