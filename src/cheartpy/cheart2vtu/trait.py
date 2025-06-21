from __future__ import annotations

import abc
import enum
from typing import TYPE_CHECKING

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
