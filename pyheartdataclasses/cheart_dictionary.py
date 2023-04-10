#!/usr/bin/env python3
from typing import Final


# Solver Algorithms
class OPTIONS_ALGORITHM():
  seq_fp_linesearch : Final = "seq_fp_linesearch"
  SOLVER_SEQUENTIAL : Final = "SOLVER_SEQUENTIAL"

class OPTIONS_BASIS():
  pass
# Solver Group Options

class OPTIONS_SG():
  AddVariables : Final = "AddVariables"
  export_initial_condition : Final = "export_initial_condition"
  L2TOL : Final = "L2TOL"
  L2PERCENT : Final = "L2PERCENT"
  INFRES : Final = "INFRES"
  INFUPDATE : Final = "INFUPDATE"
  INFDEL : Final = "INFDEL"
  ITERATION : Final = "ITERATION"
  SUBITERATION : Final = "SUBITERATION"
  LINESEARCHITER : Final = "LINESEARCHITER"
  SUBITERFRACTION : Final = "SUBITERFRACTION"
  INFRELUPDATE : Final = "INFRELUPDATE"
  L2RESRELPERCENT : Final = "L2RESRELPERCENT"

# Variable Settings
class OPTIONS_VARIABLE():
  INIT_EXPR : Final = "INIT_EXPR"
  TEMPORAL_UPDATE_EXPR : Final = "TEMPORAL_UPDATE_EXPR"
  TEMPORAL_UPDATE_FILE : Final = "TEMPORAL_UPDATE_FILE"
  ReadBinary : Final = "ReadBinary"
  ReadMMap : Final = "ReadMMap"

# Solvers
class OPTIONS_SOLVER():
  SOLVER_MUMPS : Final = "SOLVER_MUMPS"

# Solid Problems
class OPTIONS_SOLIDPROBLEM():
  transient_elasticity : Final = "transient_elasticity"
  quasi_static_elasticity : Final = "quasi_static_elasticity"

class OPTIONS_L2PROJECTION():
  l2solidprojection_problem : Final = 'l2solidprojection_problem'
  Solid_Master_Override : Final = 'Solid-Master-Override'

class OPTIONS_PROBLEMS():
  SOLID=OPTIONS_SOLIDPROBLEM()
  L2=OPTIONS_L2PROJECTION()
  norm_calculation : Final = 'norm_calculation'

# Element types
class OPTIONS_ELEMENT():
  HEXAHEDRAL_ELEMENT : Final = "HEXAHEDRAL_ELEMENT"

# Topology Settings

class OPTIONS_TOPOLOGY():
  EmbeddedInTopology : Final = "EmbeddedInTopology"

# Boundary Conditions
class OPTIONS_BC():
  Dirichlet : Final = "Dirichlet"

class OPTIONS_MATLAWS():
  neohookean : Final = 'neohookean'

class CHEART_OPTS():
  SG=OPTIONS_SG()
  B=OPTIONS_BASIS()
  T=OPTIONS_TOPOLOGY()
  E=OPTIONS_ELEMENT()
  V=OPTIONS_VARIABLE()
  P=OPTIONS_PROBLEMS()
  L=OPTIONS_MATLAWS()
  BC=OPTIONS_BC()
  SOL=OPTIONS_SOLVER()
  ALG=OPTIONS_ALGORITHM()

OPT=CHEART_OPTS()