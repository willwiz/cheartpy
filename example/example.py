#!/usr/bin/env python3
from ..src.cheartpy.dataclass.cheart_dataclass import *

def get_PFile():
  p=PFile(h=
"""
This is a helpful message.
It really is.
""",
  output_path='pfiles')

  time = TimeScheme('time', 1, 10, 0.005)
  p.AddTimeScheme(time)

  b1 = Basis("LinBasis", HEXAHEDRAL_ELEMENT, "NODAL_LAGRANGE1", "GAUSS_LEGENDRE9" )
  b2 = Basis("QuadBasis", HEXAHEDRAL_ELEMENT, "NODAL_LAGRANGE2", "GAUSS_LEGENDRE9" )
  p.AddBasis(b1, b2)


  t1 = Topology("TP1", 'mesh/cube10_lin', 'LinBasis')
  t2 = Topology("TP2", 'mesh/cube10_quad', 'QuadBasis')
  p.AddTopology(t1,t2)
  p.AddInterface(TopInterface('p2p1', "OneToOne", ["TP1","TP2"]))

  space = Variable("Space", t2, 3, file='mesh/cube10_quad')
  disp  = Variable("Disp", "TP2", 3)
  pres  = Variable("Pres", "TP1", 1)
  p.AddVariable(space, disp, pres)
  p.SetExportFrequency("Space", disp, "Pres", freq=1)

  mp = SolidProblem("Solid", quasi_static_elasticity)
  mp.UseVariable("Space", space)
  mp.UseVariable("Displacement", disp)
  mp.UseVariable("Pressure", pres)
  mp.UseOption("Perturbation-scale", 1e-9)
  mp.UseOption("Density", 1.0e-6)
  mp.UseOption("SetProblemTimeDiscretization", "time_scheme_backward_euler", "backward")

  mp.AddMatlaw(Matlaw('neohookean', [0.5]))

  left = Expression('still', ['0', '0', '0'])
  right = Expression('move', ['t', '0', '0'])
  p.AddExpression(left, right)

  bc = BoundaryCondition()
  bc.AddPatch(BCPatch(1, 'Disp', 'Dirichlet', 'still'))
  bc.AddPatch(BCPatch(2, 'Disp', 'Dirichlet', 'move'))  # fix val could be float or expr
  mp.BC = bc

  p.AddProblem(mp)

  mat = SolverMatrix("SolidMatrix", SOLVER_MUMPS, [mp])
  mat.AddSetting("ordering", "parallel")
  mat.AddSetting("SuppressOutput")
  mat.AddSetting("SolverMatrixCalculation", "evaluate_every_build")
  p.AddMatrix(mat)

  g=SolverGroup("Main",time)
  p.AddSolverGroup(g)
  g.AddSetting("L2TOL", "1e-9")
  g.AddSetting("L2PERCENT", "1e-10")
  g.export_initial_condition=True
  sg2 = SolverSubGroup('2', seq_fp_linesearch, ["SolidMatrix"])
  g.AddSolverSubGroup(sg2)

  return p


def main():
  p = get_PFile()
  name="test.P"
  with open(name,'w') as f:
    p.write(f)

if __name__=="__main__":
  main()
