import abc
from typing import TYPE_CHECKING, Literal, Self, TextIO

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence, ValuesView

    from cheartpy.fe.aliases import (
        VARIABLE_UPDATE_SETTING,
        CheartBasisType,
        CheartElementType,
        CheartQuadratureType,
        CheartTopologySetting,
    )

__all__ = [
    "BC_VALUE",
    "EXPRESSION_VALUE",
    "IBCPatch",
    "IBasis",
    "IBoundaryCondition",
    "ICheartBasis",
    "ICheartTopology",
    "IDataInterp",
    "IDataPointer",
    "IExpression",
    "ILaw",
    "IProblem",
    "IQuadrature",
    "ITimeScheme",
    "ITopInterface",
    "IVariable",
]

type EXPRESSION_VALUE = (
    int
    | str
    | float
    | IVariable
    | IExpression
    | IDataInterp
    | tuple[IVariable, int]
    | tuple[IExpression, int]
    | tuple[IDataInterp, int]
)

type BC_VALUE = IExpression | IVariable | str | int | float | tuple[IVariable, int]


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
    def __getitem__[T: int | None](self, key: T) -> tuple[IExpression, T]: ...
    @abc.abstractmethod
    def idx(self, key: int) -> str: ...
    @abc.abstractmethod
    def get_values(
        self,
    ) -> Sequence[EXPRESSION_VALUE]: ...
    @abc.abstractmethod
    def add_deps(self, *var: IExpression | IVariable | None) -> None: ...
    @abc.abstractmethod
    def add_expr_deps(self, *var: IExpression) -> None: ...
    @abc.abstractmethod
    def get_expr_deps(
        self,
    ) -> ValuesView[IExpression]: ...
    @abc.abstractmethod
    def add_var_deps(self, *var: IVariable) -> None: ...
    @abc.abstractmethod
    def get_var_deps(
        self,
    ) -> ValuesView[IVariable]: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class IBasis(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def order(self) -> Literal[0, 1, 2]: ...
    @property
    @abc.abstractmethod
    def kind(self) -> CheartBasisType: ...


class IQuadrature(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def gp(self) -> int: ...
    @property
    @abc.abstractmethod
    def kind(self) -> CheartQuadratureType: ...


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
    def order(self) -> Literal[0, 1, 2]: ...
    @property
    @abc.abstractmethod
    def gp(self) -> int: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class ICheartTopology(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def __bool__(self) -> bool: ...
    @property
    @abc.abstractmethod
    def order(self) -> Literal[0, 1, 2] | None: ...
    @property
    @abc.abstractmethod
    def mesh(self) -> str | None: ...
    @property
    @abc.abstractmethod
    def discontinuous(self) -> bool: ...
    @discontinuous.setter
    @abc.abstractmethod
    def discontinuous(self, val: bool) -> None: ...
    @abc.abstractmethod
    def get_basis(self) -> ICheartBasis | None: ...
    @abc.abstractmethod
    def add_setting(
        self,
        task: CheartTopologySetting,
        val: int | tuple[ICheartTopology, int] | None = None,
    ) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class ITopInterface(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @abc.abstractmethod
    def __hash__(self) -> int: ...
    @property
    @abc.abstractmethod
    def method(self) -> Literal["OneToOne", "ManyToOne"]: ...
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
    @abc.abstractmethod
    def __bool__(self) -> Literal[True]: ...
    @property
    @abc.abstractmethod
    def order(self) -> Literal[0, 1, 2] | None: ...
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
    def add_setting(
        self,
        task: VARIABLE_UPDATE_SETTING,
        val: str | IExpression,
    ) -> None: ...
    @abc.abstractmethod
    def set_format(self, fmt: Literal["TXT", "BINARY", "MMAP"]) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class IBCPatch:
    @abc.abstractmethod
    def __hash__(self) -> int: ...
    @abc.abstractmethod
    def use_option(self) -> None: ...
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
    def add_patch(self, *patch: IBCPatch) -> None: ...
    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...


class IProblem(abc.ABC):
    @abc.abstractmethod
    def __repr__(self) -> str: ...
    @property
    @abc.abstractmethod
    def buffering(self) -> bool: ...
    @buffering.setter
    @abc.abstractmethod
    def buffering(self, val: bool) -> None: ...
    @abc.abstractmethod
    def get_prob_vars(self) -> Mapping[str, IVariable]: ...
    @abc.abstractmethod
    def add_deps(self, *var: IVariable | IExpression | None) -> None: ...
    @abc.abstractmethod
    def get_var_deps(self) -> ValuesView[IVariable]: ...
    @abc.abstractmethod
    def add_var_deps(self, *var: IVariable | None) -> None: ...
    @abc.abstractmethod
    def get_expr_deps(self) -> ValuesView[IExpression]: ...
    @abc.abstractmethod
    def add_expr_deps(self, *expr: IExpression | None) -> None: ...
    @abc.abstractmethod
    def add_state_variable(self, *var: IVariable | IExpression | None) -> None: ...
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
    def add_var_deps(self, *var: IVariable) -> None: ...
    @abc.abstractmethod
    def get_var_deps(self) -> ValuesView[IVariable]: ...
    @abc.abstractmethod
    def add_expr_deps(self, *expr: IExpression) -> None: ...
    @abc.abstractmethod
    def get_expr_deps(self) -> ValuesView[IExpression]: ...
