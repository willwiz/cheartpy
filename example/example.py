#!/usr/bin/env python3

from cheartpy.cheart_core.aliases import *  # type: ignore
from cheartpy.cheart_core.p_file import PFile
from cheartpy.cheart_core.api import (
    create_basis,
    create_solver_group,
    create_solver_matrix,
    create_solver_subgroup,
    create_time_scheme,
    create_topology,
    create_variable,
)
from cheartpy.cheart_core.physics.solid_mechanics.solid_problems import (
    create_solid_problem,
    Matlaw,
)
from cheartpy.cheart_core.base_types.expressions import Expression
from cheartpy.cheart_core.base_types.problems import BCPatch


def get_PFile():
    p = PFile(
        h="""
This is a helpful message.
It really is.
""",
        output_path="pfiles",
    )

    mesh = "mesh/cube10"

    time = create_time_scheme("time", 1, 10, 0.005)
    b1 = create_basis(
        "LinBasis", "HEXAHEDRAL_ELEMENT", "NODAL_LAGRANGE", "GAUSS_LEGENDRE", 1, 1
    )
    b2 = create_basis(
        "QuadBasis", "HEXAHEDRAL_ELEMENT", "NODAL_LAGRANGE", "GAUSS_LEGENDRE", 2, 3
    )
    t1 = create_topology("TP1", b1, mesh + "_lin")
    t2 = create_topology("TP2", b2, mesh + "_quad")
    p.AddInterface("ManyToOne", [t2, t1])

    space = create_variable("Space", t2, 3, space="mesh/cube10_quad")
    disp = create_variable("Disp", t2, 3)
    pres = create_variable("Pres", t1, 1)
    p.SetExportFrequency(space, disp, pres, freq=1)

    mp = create_solid_problem("Solid", "QUASI_STATIC", space, disp, pres=pres)
    mp.UseOption("Perturbation-scale", 1.0e-6)
    mp.UseOption("Density", 1.0e-6)
    mp.UseOption(
        "SetProblemTimeDiscretization", "time_scheme_backward_euler", "backward"
    )
    mp.AddMatlaw(Matlaw("neohookean", [0.5]))

    left = Expression("still", [0])
    right = Expression("move", ["t", "0", "0"])
    mp.bc.AddPatch(BCPatch(1, disp[1], "dirichlet", left))
    mp.bc.AddPatch(BCPatch(2, disp, "dirichlet", right))

    mat = create_solver_matrix("SolidMatrix", "SOLVER_MUMPS", mp)
    mat.AddSetting("ordering", "parallel")
    mat.AddSetting("SuppressOutput")
    mat.AddSetting("SolverMatrixCalculation", "evaluate_every_build")

    g = create_solver_group("Main", time)
    p.AddSolverGroup(g)
    g.AddSetting("L2TOL", "1e-9")
    g.AddSetting("L2PERCENT", "1e-10")
    g.export_initial_condition = True
    sg2 = create_solver_subgroup("seq_fp_linesearch", mat)
    g.AddSolverSubGroup(sg2)

    return p


def main():
    p = get_PFile()
    name = "test.P"
    with open(name, "w") as f:
        p.write(f)


if __name__ == "__main__":
    main()
