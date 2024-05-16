import abc
from typing import TextIO, Self
from ..aliases import *

__all__ = [
    "ExpressionValue",
    "_TimeScheme",
    "_DataPointer",
    "_DataInterp",
    "_Expression",
    "_Basis",
    "_Quadrature",
    "_CheartBasis",
    "_CheartTopology",
    "_TopInterface",
    "_Variable",
    "_BCPatch",
    "_BoundaryCondition",
    "_Problem",
]

type ExpressionValue = str | float | "_DataInterp" | "_Variable" | "_Expression"


class _TimeScheme(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class _DataPointer(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class _DataInterp(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...


class _Expression(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def __getitem__[T: int | None](self, key: T) -> tuple["_Expression", T]: ...
    @abc.abstractmethod
    def idx(self, key: int) -> str: ...
    @abc.abstractmethod
    def get_values(
        self,
    ) -> list[ExpressionValue | tuple[ExpressionValue, int]]: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class _Basis(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...


class _Quadrature(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...


class _CheartBasis(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class _CheartTopology(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
    @abc.abstractmethod
    def AddSetting(
        self,
        task: CheartTopologySetting,
        val: int | tuple["_CheartTopology", int] | None = None,
    ): ...


class _TopInterface(abc.ABC):
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class _Variable(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def __getitem__[T: int | None](self, key: T) -> tuple[Self, T]: ...
    @abc.abstractmethod
    def get_data(self) -> str | None: ...
    @abc.abstractmethod
    def get_top(self) -> list[_CheartTopology]: ...
    @abc.abstractmethod
    def get_expressions(self) -> list[_Expression]: ...
    @abc.abstractmethod
    def get_export_frequency(self) -> int: ...
    @abc.abstractmethod
    def idx(self, key: int) -> str: ...
    @abc.abstractmethod
    def AddSetting(
        self,
        task: Literal[
            "INIT_EXPR",
            "TEMPORAL_UPDATE_EXPR",
            "TEMPORAL_UPDATE_FILE",
            "TEMPORAL_UPDATE_FILE_LOOP",
        ],
        val: str | _Expression,
    ): ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class _BCPatch:
    @abc.abstractmethod
    def UseOption(self) -> None: ...
    @abc.abstractmethod
    def get_values(self) -> list[_Expression | _Variable | str | int | float]: ...
    @abc.abstractmethod
    def string(self) -> str: ...


class _BoundaryCondition(abc.ABC):
    @abc.abstractmethod
    def get_patches(self) -> list[_BCPatch] | None: ...
    @abc.abstractmethod
    def AddPatch(self, *patch: _BCPatch) -> None: ...
    @abc.abstractmethod
    def DefPatch(
        self,
        id: int,
        component: _Variable,
        type: BoundaryType,
        *val: _Expression | str | int | float,
    ) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class _Problem(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def get_variables(self) -> dict[str, _Variable]: ...
    @abc.abstractmethod
    def get_aux_vars(self) -> dict[str, _Variable]: ...
    @abc.abstractmethod
    def get_bc_patches(self) -> list[_BCPatch]: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
