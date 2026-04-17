import enum
from collections.abc import Collection, Mapping
from typing import Literal

BoundaryType = Literal[
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


SolidProblemType = Literal[
    "TRANSIENT",
    "QUASI_STATIC",
]


class SolidProblemEnum(enum.StrEnum):
    TRANSIENT = "transient_elasticity"
    QUASI_STATIC = "quasi_static_elasticity"


SolidVariableValue = Literal[
    "Space",
    "Displacement",
    "Velocity",
    "Pressure",
    "Fibers",
    "GenStruc",
    "ResidualF",
    "ResidualF-weights",
    "ResidualF-vectors",
    "ResidualF-deformedvectors",
]


class SolidVariableEnum(enum.StrEnum):
    Space = "Space"
    Disp = "Disp"
    Velocity = "Velocity"
    Pressure = "Pressure"
    Fibers = "Fibers"
    GenStruc = "GenStruc"
    ResidualF = "ResidualF"
    ResidualF_weights = "ResidualF-weights"
    ResidualF_vectors = "ResidualF-vectors"
    ResidualF_deformedvectors = "ResidualF-deformedvectors"


SolidOption = Literal[
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


SolidFlag = Literal["Inverse-mechanics", "No-buffering"]


class SolidFlagEnum(enum.StrEnum):
    Inverse_mechanics = "Inverse-mechanics"
    No_buffering = "No-buffering"


L2SolidCalculationType = Literal["cauchy_stress", "deformation_gradient"]


class L2SolidCalculationEnum(enum.StrEnum):
    cauchy_stress = "cauchy_stress"
    deformation_gradient = "deformation_gradient"


CartesianRigidBodyConstraint = Mapping[Literal["T", "R"], Collection[Literal["x", "y", "z"]]]
BoundaryRigidBodyConstraint = Mapping[Literal["T", "R"], Collection[Literal["n", "p"]]]

type RigidBodyConstraint = CartesianRigidBodyConstraint | BoundaryRigidBodyConstraint
