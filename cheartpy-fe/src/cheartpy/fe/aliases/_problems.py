import enum
from typing import Literal

type BoundaryType = Literal[
    "dirichlet",
    "neumann",
    "neumann_ref",
    "neumann_nl",
    "stabilized_neumann",
    "consistent",
    "scaled_normal",
    "scaled_normal_ref",
]


class BoundaryEnum(enum.StrEnum):
    dirichlet = "dirichlet"
    neumann = "neumann"
    neumann_ref = "neumann_ref"
    neumann_nl = "neumann_nl"
    stabilized_neumann = "stabilized_neumann"
    consistent = "consistent"
    scaled_normal = "scaled_normal"
    scaled_normal_ref = "scaled_normal_ref"


type SolidProblemType = Literal[
    "TRANSIENT",
    "QUASI_STATIC",
]


class SolidProblemEnum(enum.StrEnum):
    TRANSIENT = "transient_elasticity"
    QUASI_STATIC = "quasi_static_elasticity"


type SolidVariable = Literal[
    "Space",
    "Displacement",
    "Velocity",
    "Pressure",
    "Fibers",
    "GenStruc",
]


class SolidVariableEnum(enum.StrEnum):
    Space = "Space"
    Disp = "Disp"
    Velocity = "Velocity"
    Pressure = "Pressure"
    Fibers = "Fibers"
    GenStruc = "GenStruc"


type SolidOption = Literal[
    "Density",
    "Perturbation-scale",
    "SetProblemTimeDiscretization",
    "UseStabilization",
]


class SolidOptionEnum(enum.StrEnum):
    Density = "Density"
    Perturbation_scale = "Perturbation-scale"
    SetProblemTimeDiscretization = "SetProblemTimeDiscretization"
    UseStabilization = "UseStabilization"


type SolidFlag = Literal["Inverse-mechanics", "No-buffering"]


class SolidFlagEnum(enum.StrEnum):
    Inverse_mechanics = "Inverse-mechanics"
    No_buffering = "No-buffering"


type L2SolidCalculationType = Literal["cauchy_stress", "deformation_gradient"]


class L2SolidCalculationEnum(enum.StrEnum):
    cauchy_stress = "cauchy_stress"
    deformation_gradient = "deformation_gradient"
