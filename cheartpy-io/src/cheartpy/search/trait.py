import abc
import dataclasses as dc
import enum
import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal, override

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
    def __iter__(self) -> Iterator[int | tuple[int, int]]: ...
    @abc.abstractmethod
    def __len__(self) -> int: ...
    @property
    @abc.abstractmethod
    def mode(self) -> ProgramMode: ...


class _VariableType(abc.ABC):
    @abc.abstractmethod
    def __getitem__(self, time: int | tuple[int, int]) -> Path: ...

    @abc.abstractmethod
    def __str__(self) -> str: ...

    @property
    @abc.abstractmethod
    def ext(self) -> str: ...


class StaticFile(_VariableType):
    __slots__ = ["_file"]

    _file: Path

    def __init__(self, file: Path | str) -> None:
        self._file = Path(file)

    @override
    def __getitem__(self, time: int | tuple[int, int]) -> Path:
        return self._file

    @override
    def __str__(self) -> str:
        return re.split(r"[-.]", self._file.stem)[0]

    @property
    @override
    def ext(self) -> str:
        return self._file.suffix

    @property
    def is_dynamic(self) -> Literal[False]:
        return False


class DynamicFile(_VariableType):
    __slots__ = ["_ext", "_folder", "_var"]

    _folder: Path
    _var: str
    _ext: str

    def __init__(self, folder: Path | str, var: str, ext: str) -> None:
        self._folder = Path(folder)
        self._var = var
        self._ext = ext

    @override
    def __getitem__(self, time: int | tuple[int, int]) -> Path:
        match time:
            case int():
                return self._folder / f"{self._var}-{time}{self._ext}"
            case i, j:
                return self._folder / f"{self._var}-{i}.{j}{self._ext}"

    @override
    def __str__(self) -> str:
        return f"{self._var}"

    @property
    @override
    def ext(self) -> str:
        return self._ext

    @property
    def is_dynamic(self) -> Literal[True]:
        return True


type FileType = StaticFile | DynamicFile


@dc.dataclass(slots=True, frozen=True)
class FileVariable:
    fname: FileType
    indices: set[int]
    subindices: set[tuple[int, int]]

    def __str__(self) -> str:
        return str(self.fname)
