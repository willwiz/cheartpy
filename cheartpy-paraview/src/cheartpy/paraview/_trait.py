import enum
from collections.abc import Sequence
from typing import ReadOnly, TypedDict


class ProgramMode(enum.StrEnum):
    none = "none"
    search = "search"
    searchsubindex = "searchsubindex"
    range = "range"
    subindex = "subindex"
    subauto = "subauto"


class _TimeSeriesItem(TypedDict):
    name: str
    time: float


TimeSerie = TypedDict(
    "TimeSerie",
    {
        "file-series-version": ReadOnly[str],
        "files": ReadOnly[Sequence[_TimeSeriesItem]],
    },
)
