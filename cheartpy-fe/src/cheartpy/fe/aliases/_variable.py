import enum
from typing import Literal

VARIABLE_EXPORT_FORMAT = Literal["TXT", "BINARY", "MMAP"]


class VariableExportFormat(enum.StrEnum):
    TXT = "TXT"
    BINARY = "ReadBinary"
    MMAP = "ReadMMap"


VARIABLE_UPDATE_SETTING = Literal[
    "INIT_EXPR",
    "TEMPORAL_UPDATE_EXPR",
    "TEMPORAL_UPDATE_FILE",
    "TEMPORAL_UPDATE_FILE_LOOP",
]


class VariableUpdateSetting(enum.StrEnum):
    INIT_EXPR = "INIT_EXPR"
    TEMPORAL_UPDATE_EXPR = "TEMPORAL_UPDATE_EXPR"
    TEMPORAL_UPDATE_FILE = "TEMPORAL_UPDATE_FILE"
    TEMPORAL_UPDATE_FILE_LOOP = "TEMPORAL_UPDATE_FILE_LOOP"
