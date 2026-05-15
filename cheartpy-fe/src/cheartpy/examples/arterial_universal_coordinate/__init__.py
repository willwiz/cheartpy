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
from pydantic import BaseModel
from pytools.result import Err, Ok, Result
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from cheartpy.fe.aliases import TopologyDef
    from cheartpy.fe.trait import IBCPatch, ICheartTopology, IPFile, IVariable

TX = Literal["X"]


class VariableNameKwargs(TypedDict, total=False):
    a_z: str
    a_r: str
    v_z: str
    v_r: str


class BoundaryValueKwargs(TypedDict, total=True):
    z: Mapping[int, float]
    r: Mapping[int, float]


class APIKwargs[T](TypedDict, total=False):
    top: Required[TopologyDef[T]]
    bc: Required[BoundaryValueKwargs]
    prefix: VariableNameKwargs
    output_dir: Path


@dc.dataclass(slots=True)
class VariableList:
    space: IVariable
    a_z: IVariable
    a_r: IVariable
    v_z: IVariable
    v_r: IVariable


class Options(BaseModel): ...


def read_options[T](**kwargs: Unpack[APIKwargs[T]]) -> Options: ...


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
        space=space,
        a_z=create_variable(prefix.get("a_z", "Az"), t, 1),
        a_r=create_variable(prefix.get("a_r", "Ar"), t, 1),
        v_z=create_variable(prefix.get("v_z", "Vz"), t, space.get_dim()),
        v_r=create_variable(prefix.get("v_r", "Vr"), t, space.get_dim()),
    )
    return Ok(vs)


def create_bc_patches[T](
    vlist: VariableList, **kwargs: Unpack[APIKwargs[T]]
) -> dict[str, list[IBCPatch]]:
    return {
        f"{vlist.a_z!s}": [
            create_bcpatch(i, vlist.a_z, "dirichlet", v) for i, v in kwargs["bc"]["z"].items()
        ],
        f"{vlist.a_r!s}": [
            create_bcpatch(i, vlist.a_r, "dirichlet", v) for i, v in kwargs["bc"]["r"].items()
        ],
    }


def uac_pfile[T](**kwargs: Unpack[APIKwargs[T]]) -> IPFile:
    time = create_time_scheme("time", 1, 1, 1)
    top, iface = create_problem_topology(**kwargs)
    var = create_variable_list(top["X"], **kwargs).unwrap()
    bcs = create_bc_patches(var, **kwargs)
    coord_probs = {
        f"{v!s}": create_laplace_problem(f"Problem{v!s}", var.space, v) for v in (var.a_r, var.a_z)
    }

    for k, p in coord_probs.items():
        p.bc.add_patch(*bcs[k])
    vec_probs = {
        f"{v!s}": create_l2varprojection_problem(f"Problem{v!s}", var.space, a, v, "gradient")
        for a, v in ((var.a_r, var.v_r), (var.a_z, var.v_z))
    }
    coord_matrices = {
        k: create_solver_matrix(f"Matrix{k!s}", "SOLVER_MUMPS", p) for k, p in coord_probs.items()
    }
    vec_matrices = {
        k: create_solver_matrix(f"Matrix{k!s}", "SOLVER_MUMPS", p) for k, p in vec_probs.items()
    }

    solver_subgroups = {
        k: create_solver_subgroup("seq_fp_linesearch", m) for k, m in coord_matrices.items()
    } | {k: create_solver_subgroup("SOLVER_SEQUENTIAL", m) for k, m in vec_matrices.items()}
    sg = create_solver_group("Main", time, *solver_subgroups.values())
    pfile = create_pfile()
    pfile.add_solvergroup(sg)
    if output_path := kwargs.get("output_dir"):
        pfile.set_outputpath(output_path)
    pfile.add_interface(*iface)
    return pfile
