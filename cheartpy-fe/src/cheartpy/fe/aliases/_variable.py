import enum
from typing import Literal

VariableExportFormat = Literal["TXT", "BINARY", "MMAP"]


class VariableExportEnum(enum.StrEnum):
    TXT = "TXT"
    BINARY = "ReadBinary"
    MMAP = "ReadMMap"


VariableUpdateSetting = Literal[
    "INIT_EXPR",
    "TEMPORAL_UPDATE_EXPR",
    "TEMPORAL_UPDATE_FILE",
    "TEMPORAL_UPDATE_FILE_LOOP",
]


class VariableUpdateEnum(enum.StrEnum):
    INIT_EXPR = "INIT_EXPR"
    TEMPORAL_UPDATE_EXPR = "TEMPORAL_UPDATE_EXPR"
    TEMPORAL_UPDATE_FILE = "TEMPORAL_UPDATE_FILE"
    TEMPORAL_UPDATE_FILE_LOOP = "TEMPORAL_UPDATE_FILE_LOOP"
