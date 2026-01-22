import abc
import enum
from collections.abc import Sequence
from typing import TYPE_CHECKING, ReadOnly, TypedDict

if TYPE_CHECKING:
    from pathlib import Path


class ProgramMode(enum.StrEnum):
    none = "none"
    search = "search"
    searchsubindex = "searchsubindex"
    range = "range"
    subindex = "subindex"
    subauto = "subauto"


class IFormattedName(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...
    @abc.abstractmethod
    def __getitem__(self, i: str | int) -> Path: ...


class _TimeSeriesItem(TypedDict):
    name: str
    time: float


TIME_SERIES = TypedDict(
    "TIME_SERIES",
    {
        "file-series-version": ReadOnly[str],
        "files": ReadOnly[Sequence[_TimeSeriesItem]],
    },
)
