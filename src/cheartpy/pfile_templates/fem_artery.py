import os
import cheartpy.cheart_core as c
from cheartpy.cheart_core import PFile, Basis, TimeScheme, HEXAHEDRAL_ELEMENT
from cheartpy.cheart_core.basetypes import Basis, TimeScheme
from cheartpy.cheart_core.keywords import HEXAHEDRAL_ELEMENT
from cheartpy.cheart_core.data_types import PFile


def get_PFile() -> PFile:
    output_folder = f"data"
    mesh = f"mesh/artery"
    os.makedirs(output_folder, exist_ok=True)
    p = c.PFile(h="Anna's stuff", output_path=output_folder)
    # time = TimeScheme('time', 1, int(nt/6), new_time_file)
    time = TimeScheme("time", 1, 5000, 0.0005)

    b1 = Basis("LinBasis", c.EL.TETRAHEDRAL, "NODAL_LAGRANGE1", "GAUSS_LEGENDRE9")
    b2 = Basis("QuadBasis", c.EL.TETRAHEDRAL, "NODAL_LAGRANGE2", "GAUSS_LEGENDRE9")

    t1 = c.Topology("TP1", mesh + "_lin", b1)
    t2 = c.Topology("TP2", mesh + "_quad", b2)
    interface = c.TopInterface("main", "OneToOne", [t1, t2])
    p.AddInterface(interface)

    space = c.Variable("Space", t2, 3, file=mesh + "_quad")
    disp = c.Variable("Disp", t2, 3)
    pres = c.Variable("Pres", t1, 1)

    velocity = 0.1
    left = c.Expression("still", [f"{- velocity}*t", 0, 0])
    right = c.Expression("move", [f"{velocity}*t", "0", "0"])

    mp = c.SolidProblem("Solid", c.SOLID.QUASI_STATIC_ELASTICITY)
    mp.UseVariable("Space", space)
    mp.UseVariable("Displacement", disp)
    mp.UseVariable("Pressure", pres)
    mp.UseOption("Density", 1.0e-6)
    mp.UseOption(
        "SetProblemTimeDiscretization", "time_scheme_backward_euler", "backward"
    )

    neohook = c.Matlaw("neohookean", [0.2])
    mp.AddMatlaw(neohook)

    mp.BC = c.BoundaryCondition()
    mp.BC.AddPatch(c.BCPatch(5, "Disp", "dirichlet", left))
    mp.BC.AddPatch(c.BCPatch(6, "Disp", "dirichlet", right))

    mat = c.SolverMatrix("SolidMatrix", c.SOLVER.SOLVER_MUMPS, [mp])
    mat.AddSetting("ordering", "parallel")
    mat.AddSetting("SuppressOutput")
    mat.AddSetting("SolverMatrixCalculation", "evaluate_every_build")

    g = c.SolverGroup("Main", time)
    p.AddSolverGroup(g)

    sg1 = c.SolverSubGroup("solid", c.SOLVER.SEQ_FP_LINESEARCH, [mat])

    g.set_convergence(TolSettings.L2TOL, 1e-9)
    g.set_convergence("L2PERCENT", 1e-10)
    g.set_iteration("SUBITERATION", 10)
    g.AddVariable(storeR, storeL)
    g.export_initial_condition = True
    g.AddSolverSubGroup(sg1, sg2, sg3)

    p.SetExportFrequency(disp, freq=-1)
    p.SetExportFrequency(space, pres, stress, freq=-1)
    p.SetExportFrequency(fib, storeL, storeR, freq=-1)

    return p


def main():
    p = get_PFile()
    name = "test.P"
    with open(name, "w") as f:
        p.write(f)


if __name__ == "__main__":
    main()
