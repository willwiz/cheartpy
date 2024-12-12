__all__ = ["Variable2Getter", "Variable1Getter", "Variable0Getter"]
from typing import Final, Iterator
from ..tools.path_tools import path
from .traits import *


class Variable2Getter(IVariableGetter):
    __slots__ = ["_var1", "_var2", "_idx"]

    _var1: Final[str]
    _var2: Final[str]
    _idx: list[int]

    def __init__(
        self, var1: str, var2: str, idx: list[int], root: str | None = None
    ) -> None:
        self._var1 = path(root, var1)
        self._var2 = path(root, var2)
        self._idx = idx

    def __iter__(self) -> Iterator[tuple[int, str, str]]:
        for i in self._idx:
            yield i, f"{self._var1}-{i}.D", f"{self._var2}-{i}.D"


class Variable1Getter(IVariableGetter):
    __slots__ = ["_var1", "_var2", "_idx", "_reversed"]

    _var1: Final[str]
    _var2: Final[str | None]
    _idx: list[int]
    _reversed: Final[bool]

    def __init__(
        self,
        var1: str,
        var2: str | None,
        idx: list[int],
        root: str | None = None,
        reversed: bool = False,
    ) -> None:
        self._var1 = path(root, var1)
        self._var2 = path(root, var2) if var2 else None
        self._idx = idx
        self._reversed = reversed

    def __iter__(self) -> Iterator[tuple[int, str | None, str | None]]:
        if self._reversed:
            for i in self._idx:
                yield i, self._var2, f"{self._var1}-{i}.D"
        else:
            for i in self._idx:
                yield i, f"{self._var1}-{i}.D", self._var2


class Variable0Getter(IVariableGetter):
    __slots__ = ["_var1", "_var2"]

    _var1: Final[str | None]
    _var2: Final[str | None]

    def __init__(
        self, var1: str | None, var2: str | None, root: str | None = None
    ) -> None:
        self._var1 = path(root, var1) if var1 else None
        self._var2 = path(root, var2) if var2 else None

    def __iter__(self) -> Iterator[tuple[str, str | None, str | None]]:
        yield "", self._var1, self._var2
