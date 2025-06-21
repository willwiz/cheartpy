from __future__ import annotations

from pathlib import Path

__all__ = ["Variable0Getter", "Variable1Getter", "Variable2Getter"]
from typing import TYPE_CHECKING, Final

from .traits import IVariableGetter

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


class Variable2Getter(IVariableGetter):
    __slots__ = ["_idx", "_root", "_var1", "_var2"]

    _root: Final[Path]
    _var1: str
    _var2: str
    _idx: Sequence[int]

    def __init__(
        self,
        var1: str,
        var2: str,
        idx: Sequence[int],
        root: Path | str | None = None,
    ) -> None:
        self._root = Path(root) if root else Path()
        self._var1 = var1
        self._var2 = var2
        self._idx = idx

    def __iter__(self) -> Iterator[tuple[int, Path, Path]]:
        for i in self._idx:
            yield i, self._root / f"{self._var1}-{i}.D", self._root / f"{self._var2}-{i}.D"


class Variable1Getter(IVariableGetter):
    __slots__ = ["_idx", "_reversed", "_var1", "_var2"]

    _var1: Final[str | None]
    _var2: Final[str | None]
    _idx: Sequence[int]
    _reversed: Final[bool]

    def __init__(
        self,
        var1: str | None,
        var2: str | None,
        idx: Sequence[int],
        root: Path | str | None = None,
        *,
        reverse: bool = False,
    ) -> None:
        self._root = Path(root) if root else Path()
        self._var1 = var1 if var1 else None
        self._var2 = var2 if var2 else None
        # if var2 is str assume it is a constant file name
        self._idx = idx
        self._reversed = reverse

    def __iter__(self) -> Iterator[tuple[int, Path | None, Path | None]]:
        match self._var2, self._var1, self._reversed:
            case None, None, _:
                msg = "No Variable given"
                raise ValueError(msg)
            case str(v), None, _:
                for i in self._idx:
                    yield i, self._root / f"{v}-{i}.D", None
            case None, str(v), _:
                for i in self._idx:
                    yield i, None, self._root / f"{v}-{i}.D"
            case str(v1), str(v2), False:
                for i in reversed(self._idx) if self._reversed else self._idx:
                    yield i, self._root / f"{v1}-{i}.D", self._root / v2
            case str(v1), str(v2), True:
                for i in reversed(self._idx) if self._reversed else self._idx:
                    yield i, self._root / v1, self._root / f"{v2}-{i}.D"


class Variable0Getter(IVariableGetter):
    __slots__ = ["_var1", "_var2"]

    _var1: Final[Path | None]
    _var2: Final[Path | None]

    def __init__(
        self,
        var1: Path | str | None,
        var2: Path | str | None,
        root: Path | str | None = None,
    ) -> None:
        root = Path(root) if root else Path()
        self._var1 = root / var1 if var1 else None
        self._var2 = root / var2 if var2 else None

    def __iter__(self) -> Iterator[tuple[str, Path | None, Path | None]]:
        yield "", self._var1, self._var2
