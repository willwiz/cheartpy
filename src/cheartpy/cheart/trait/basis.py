import abc
from typing import Mapping, Sequence, TextIO, Self, ValuesView
from ..aliases import *

__all__ = [
    "EXPRESSION_VALUE_TYPES",
    "ITimeScheme",
    "IDataPointer",
    "IDataInterp",
    "IExpression",
    "IBasis",
    "IQuadrature",
    "ICheartBasis",
    "ICheartTopology",
    "ITopInterface",
    "IVariable",
    "IBCPatch",
    "IBoundaryCondition",
    "IProblem",
    "ILaw",
]

type EXPRESSION_VALUE_TYPES = "int|str|float|IVariable|IExpression|IDataInterp|tuple[IVariable, int]|tuple[IExpression, int]|tuple[IDataInterp, int]"


class ITimeScheme(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class IDataPointer(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class IDataInterp(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def get_datapointer(self) -> IDataPointer: ...


class IExpression(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def __len__(self) -> int: ...
    @abc.abstractmethod
    def __getitem__[T: int | None](self, key: T) -> tuple["IExpression", T]: ...
    @abc.abstractmethod
    def idx(self, key: int) -> str: ...
    @abc.abstractmethod
    def get_values(
        self,
    ) -> Sequence[EXPRESSION_VALUE_TYPES]: ...
    @abc.abstractmethod
    def add_deps(self, *vars: "IExpression|IVariable") -> None: ...
    @abc.abstractmethod
    def add_expr_deps(self, *var: "IExpression") -> None: ...
    @abc.abstractmethod
    def get_expr_deps(
        self,
    ) -> ValuesView["IExpression"]: ...
    @abc.abstractmethod
    def add_var_deps(self, *var: "IVariable") -> None: ...
    @abc.abstractmethod
    def get_var_deps(
        self,
    ) -> ValuesView["IVariable"]: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class IBasis(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def order(self) -> Literal[1, 2]: ...


class IQuadrature(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...


class ICheartBasis(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def elem(self) -> CheartElementType: ...
    @property
    @abc.abstractmethod
    def basis(self) -> IBasis: ...
    @property
    @abc.abstractmethod
    def quadrature(self) -> IQuadrature: ...
    @property
    @abc.abstractmethod
    def order(self) -> Literal[1, 2]: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class ICheartTopology(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def order(self) -> Literal[1, 2, None]: ...
    @property
    @abc.abstractmethod
    def mesh(self) -> str | None: ...
    @abc.abstractmethod
    def get_basis(self) -> ICheartBasis | None: ...
    @abc.abstractmethod
    def AddSetting(
        self,
        task: CheartTopologySetting,
        val: int | tuple["ICheartTopology", int] | None = None,
    ): ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class ITopInterface(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def __hash__(self) -> int: ...
    @abc.abstractmethod
    def get_tops(self) -> Sequence[ICheartTopology]: ...
    @abc.abstractmethod
    def get_master(self) -> ICheartTopology | None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class IVariable(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def __getitem__[T: int | None](self, key: T) -> tuple[Self, T]: ...
    @property
    @abc.abstractmethod
    def order(self) -> Literal[1, 2, None]: ...
    @abc.abstractmethod
    def add_data(self, data: str | None) -> None: ...
    @abc.abstractmethod
    def get_data(self) -> str | None: ...
    @abc.abstractmethod
    def get_top(self) -> ICheartTopology: ...
    @abc.abstractmethod
    def get_expr_deps(self) -> ValuesView[IExpression]: ...
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
        val: str | IExpression,
    ): ...
    @abc.abstractmethod
    def SetFormat(self, fmt: Literal["TXT", "BINARY", "MMAP"]) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class IBCPatch:
    @abc.abstractmethod
    def __hash__(self) -> int: ...
    @abc.abstractmethod
    def UseOption(self) -> None: ...
    @abc.abstractmethod
    def get_var_deps(self) -> ValuesView[IVariable]: ...
    @abc.abstractmethod
    def get_expr_deps(self) -> ValuesView[IExpression]: ...
    @abc.abstractmethod
    def string(self) -> str: ...


class IBoundaryCondition(abc.ABC):
    @abc.abstractmethod
    def get_vars_deps(self) -> ValuesView[IVariable]: ...
    @abc.abstractmethod
    def get_expr_deps(self) -> ValuesView[IExpression]: ...
    @abc.abstractmethod
    def get_patches(self) -> ValuesView[IBCPatch] | None: ...
    @abc.abstractmethod
    def AddPatch(self, *patch: IBCPatch) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class IProblem(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def get_prob_vars(self) -> Mapping[str, IVariable]: ...
    @abc.abstractmethod
    def get_var_deps(self) -> ValuesView[IVariable]: ...
    @abc.abstractmethod
    def add_var_deps(self, *var: IVariable) -> None: ...
    @abc.abstractmethod
    def get_expr_deps(self) -> ValuesView[IExpression]: ...
    @abc.abstractmethod
    def add_expr_deps(self, *expr: IExpression) -> None: ...
    @abc.abstractmethod
    def get_bc_patches(self) -> Sequence[IBCPatch]: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class ILaw(abc.ABC):
    @abc.abstractmethod
    def string(self) -> str: ...
    @abc.abstractmethod
    def get_prob_vars(self) -> Mapping[str, IVariable]: ...
    @abc.abstractmethod
    def add_var_deps(self, *vars: IVariable) -> None: ...
    @abc.abstractmethod
    def get_var_deps(self) -> ValuesView[IVariable]: ...
    @abc.abstractmethod
    def add_expr_deps(self, *exprs: IExpression) -> None: ...
    @abc.abstractmethod
    def get_expr_deps(self) -> ValuesView[IExpression]: ...