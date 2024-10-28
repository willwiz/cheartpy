import abc
from typing import Sequence, TextIO, Self, ValuesView
from ..aliases import *

__all__ = [
    "EXPRESSION_VALUE_TYPES",
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
    "_Law",
]

type EXPRESSION_VALUE_TYPES = "int|str|float|_Variable|_Expression|_DataInterp|tuple[_Variable, int]|tuple[_Expression, int]|tuple[_DataInterp, int]"


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
    @abc.abstractmethod
    def get_val(self) -> _DataPointer: ...


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
    ) -> Sequence[EXPRESSION_VALUE_TYPES]: ...
    @abc.abstractmethod
    def add_expr_deps(self, *var: "_Expression") -> None: ...
    @abc.abstractmethod
    def get_expr_deps(
        self,
    ) -> ValuesView["_Expression"]: ...
    @abc.abstractmethod
    def add_var_deps(self, *var: "_Variable") -> None: ...
    @abc.abstractmethod
    def get_var_deps(
        self,
    ) -> ValuesView["_Variable"]: ...
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
    @property
    @abc.abstractmethod
    def mesh(self) -> str: ...
    @abc.abstractmethod
    def get_basis(self) -> _CheartBasis | None: ...
    @abc.abstractmethod
    def AddSetting(
        self,
        task: CheartTopologySetting,
        val: int | tuple["_CheartTopology", int] | None = None,
    ): ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class _TopInterface(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def __hash__(self) -> int: ...
    @abc.abstractmethod
    def get_tops(self) -> Sequence[_CheartTopology]: ...
    @abc.abstractmethod
    def get_master(self) -> _CheartTopology | None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class _Variable(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def __getitem__[T: int | None](self, key: T) -> tuple[Self, T]: ...
    @abc.abstractmethod
    def add_data(self, data: str | None) -> None: ...
    @abc.abstractmethod
    def get_data(self) -> str | None: ...
    @abc.abstractmethod
    def get_top(self) -> _CheartTopology: ...
    @abc.abstractmethod
    def get_expressions(self) -> list[_Expression]: ...
    @abc.abstractmethod
    def get_dim(self) -> int: ...
    @abc.abstractmethod
    def get_export_frequency(self) -> int: ...
    @abc.abstractmethod
    def set_export_frequency(self, v: int) -> None: ...
    @abc.abstractmethod
    def idx(self, key: int) -> str: ...
    @abc.abstractmethod
    def AddSetting(
        self,
        task: VARIABLE_UPDATE_SETTING,
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
    def get_aux_vars(self) -> ValuesView[_Variable]: ...
    @abc.abstractmethod
    def add_aux_vars(self, *var: _Variable) -> None: ...
    @abc.abstractmethod
    def get_aux_expr(self) -> dict[str, _Expression]: ...
    @abc.abstractmethod
    def add_aux_expr(self, *expr: _Expression) -> None: ...
    @abc.abstractmethod
    def get_bc_patches(self) -> list[_BCPatch]: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class _Law(abc.ABC):
    @abc.abstractmethod
    def string(self) -> str: ...
    @abc.abstractmethod
    def get_aux_vars(self) -> dict[str, _Variable]: ...
