import abc
import enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator


__all__ = ["AUTO", "IIndexIterator", "ProgramMode", "SearchMode"]


class ProgramMode(enum.StrEnum):
    none = "none"
    search = "search"
    searchsubindex = "searchsubindex"
    range = "range"
    subindex = "subindex"
    subauto = "subauto"


class SearchMode(enum.Enum):
    none = 0
    auto = 1


AUTO = SearchMode.auto


class IIndexIterator(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[str | int]: ...
    @abc.abstractmethod
    def __len__(self) -> int: ...
    @property
    @abc.abstractmethod
    def mode(self) -> ProgramMode: ...
