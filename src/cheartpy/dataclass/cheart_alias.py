import enum


class TolSettings(enum.StrEnum):
    L2TOL = "L2TOL"
    L2PERCENT = enum.auto()
    INFRES = enum.auto()
    INFUPDATE = enum.auto()
    INFDEL = enum.auto()
    INFRELUPDATE = enum.auto()
    L2RESRELPERCENT = enum.auto()


class IterationSettings(enum.StrEnum):
    ITERATION = enum.auto()
    SUBITERATION = enum.auto()
    LINESEARCHITER = enum.auto()
    SUBITERFRACTION = enum.auto()
    GroupIterations = "GroupIterations"


class InterfaceTypes(enum.StrEnum):
    OneToOne = "OneToOne"
    ManyToOne = "ManyToOne"


class VariableExportFormat(enum.StrEnum):
    TXT = "TXT"
    BINARY = "ReadBinary"
    MMAP = "ReadMMap"
