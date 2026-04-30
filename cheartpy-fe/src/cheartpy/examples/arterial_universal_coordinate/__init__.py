import dataclasses as dc
from pathlib import Path
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
from cheartpy.fe.physics.api import create_laplace_problem
from pydantic import BaseModel
from pytools.result import Err, Ok, Result
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

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


class APIKwargs(TypedDict, total=False):
    top: Required[TopologyDef[Literal["X"]]]
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


def read_options(**kwargs: Unpack[APIKwargs]) -> Options: ...


def create_problem_topology(**kwargs: Unpack[APIKwargs]) -> CompiledTopologies[TX]:
    defn: Mapping[TX, TopologyDef[TX]] = {"X": kwargs["top"]}
    return create_topologies(defn)


def create_variable_list(t: ICheartTopology, **kwargs: Unpack[APIKwargs]) -> Result[VariableList]:
    if t.mesh is None:
        msg = f"Topology {t!s} does not have a mesh file specified."
        return Err(ValueError(msg))
    space = create_variable("X", t, 3, data=t.mesh)
    prefix = kwargs.get("prefix") or {}
    vs = VariableList(
        space=space,
        a_z=create_variable(prefix.get("a_z", "a_z"), t, 1),
        a_r=create_variable(prefix.get("a_r", "a_r"), t, 1),
        v_z=create_variable(prefix.get("v_z", "v_z"), t, space.get_dim()),
        v_r=create_variable(prefix.get("v_r", "v_r"), t, space.get_dim()),
    )
    return Ok(vs)


def create_bc_patches(
    vlist: VariableList, **kwargs: Unpack[APIKwargs]
) -> dict[str, list[IBCPatch]]:
    return {
        f"{vlist.a_z!s}": [
            create_bcpatch(i, vlist.a_z, "dirichlet", v) for i, v in kwargs["bc"]["z"].items()
        ],
        f"{vlist.a_r!s}": [
            create_bcpatch(i, vlist.a_r, "dirichlet", v) for i, v in kwargs["bc"]["r"].items()
        ],
    }


def uac_pfile(**kwargs: Unpack[APIKwargs]) -> IPFile:
    time = create_time_scheme("time", 1, 1, 1)
    top, iface = create_problem_topology(**kwargs)
    var = create_variable_list(top["X"], **kwargs).unwrap()
    bcs = create_bc_patches(var, **kwargs)
    coord_probs = {
        f"{v!s}": create_laplace_problem(f"Problem{v!s}", var.space, v) for v in (var.a_r, var.a_z)
    }
    for k, p in coord_probs.items():
        p.bc.add_patch(*bcs[k])
    matrices = {
        k: create_solver_matrix(f"Matrix{k!s}", "SOLVER_MUMPS", p) for k, p in coord_probs.items()
    }
    solver_subgroups = {
        k: create_solver_subgroup("seq_fp_linesearch", m) for k, m in matrices.items()
    }
    sg = create_solver_group("sg", time, *solver_subgroups.values())
    pfile = create_pfile()
    pfile.add_solvergroup(sg)
    pfile.set_outputpath(kwargs.get("output_dir") or Path.cwd())
    pfile.add_interface(*iface)
    return pfile
