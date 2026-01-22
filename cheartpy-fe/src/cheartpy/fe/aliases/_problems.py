import enum
from typing import Literal

BOUNDARY_TYPE = Literal[
    "dirichlet",
    "neumann",
    "neumann_ref",
    "neumann_nl",
    "stabilized_neumann",
    "consistent",
    "scaled_normal",
    "scaled_normal_ref",
]


class BoundaryType(enum.StrEnum):
    dirichlet = "dirichlet"
    neumann = "neumann"
    neumann_ref = "neumann_ref"
    neumann_nl = "neumann_nl"
    stabilized_neumann = "stabilized_neumann"
    consistent = "consistent"
    scaled_normal = "scaled_normal"
    scaled_normal_ref = "scaled_normal_ref"


SOLID_PROBLEM_TYPE = Literal[
    "TRANSIENT",
    "QUASI_STATIC",
]


class SolidProblemType(enum.StrEnum):
    TRANSIENT = "transient_elasticity"
    QUASI_STATIC = "quasi_static_elasticity"


SOLID_VARIABLES = Literal[
    "Space",
    "Displacement",
    "Velocity",
    "Pressure",
    "Fibers",
    "GenStruc",
]


class SolidVariables(enum.StrEnum):
    Space = "Space"
    Disp = "Disp"
    Velocity = "Velocity"
    Pressure = "Pressure"
    Fibers = "Fibers"
    GenStruc = "GenStruc"


SOLID_OPTIONS = Literal[
    "Density",
    "Perturbation-scale",
    "SetProblemTimeDiscretization",
    "UseStabilization",
]


class SolidOptions(enum.StrEnum):
    Density = "Density"
    Perturbation_scale = "Perturbation-scale"
    SetProblemTimeDiscretization = "SetProblemTimeDiscretization"


SOLID_FLAGS = Literal["Inverse-mechanics", "No-buffering"]


class SolidFlags(enum.StrEnum):
    Inverse_mechanics = "Inverse-mechanics"
    No_buffering = "No-buffering"


L2_SOLID_CALCULATION_TYPE = Literal["cauchy_stress", "deformation_gradient"]


class L2SolidCalculationType(enum.StrEnum):
    cauchy_stress = "cauchy_stress"
    deformation_gradient = "deformation_gradient"
