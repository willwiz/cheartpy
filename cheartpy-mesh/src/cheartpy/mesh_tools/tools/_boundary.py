import enum


class BoundaryRelation(enum.Enum):
    """Boundary relation types."""

    NONE = enum.auto()
    SURFACE = enum.auto()
    INTERIOR = enum.auto()
