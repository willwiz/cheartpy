#!/usr/bin/env python3
from typing import Final, Literal, Union


# Solver Algorithms
SEQ_FP_LINESEARCH : Final = "seq_fp_linesearch"
SOLVER_SEQUENTIAL : Final = "SOLVER_SEQUENTIAL"
# Solver Group Options


# Variable Settings
TEMPORAL_UPDATE_EXPR : Final = "TEMPORAL_UPDATE_EXPR"
# Solvers
SOLVER_MUMPS : Final = "SOLVER_MUMPS"

# Solid Problems
TRANSIENT_ELASTICITY : Final = "transient_elasticity"
QUASI_STATIC_ELASTICITY : Final = "quasi_static_elasticity"

L2SOLIDPROJECTION_PROBLEM : Final = 'l2solidprojection_problem'

NORM_CALCULATION : Final = 'norm_calculation'


# Element types
HEXAHEDRAL_ELEMENT : Final = "HEXAHEDRAL_ELEMENT"

# Topology Settings
EMBEDDEDINTOPOLOGY : Final = "EmbeddedInTopology"

# Boundary Conditions
DIRICHLET : Final = "Dirichlet"


SOLVER_GROUP_SETTINGS=Literal["L2TOL", "L2PERCENT", "INFRES", "INFUPDATE", "INFDEL",
                    "ITERATION", "SUBITERATION", "LINESEARCHITER", "SUBITERFRACTION",
                    "INFRELUPDATE", "L2RESRELPERCENT"]