import abc
import dataclasses as dc
import enum
from typing import Union, TextIO
from ..pytools import get_enum, join_fields
from .expressions import Expression
from .variables import Variable
from ..aliases import *


class BCPatch:
    id: Union[int, str]
    component: tuple[Variable, int | None]
    bctype: BoundaryType
    value: list[Expression | Variable | str | int | float]
    options: list[str | int | float]

    def __init__(
        self,
        i: int,
        component: Variable | tuple[Variable, int | None],
        bctype: BOUNDARY_TYPE | BoundaryType,
        *val: Expression | str | int | float,
    ) -> None:
        self.id = i
        if isinstance(component, Variable):
            component, idx = component, None
        else:
            component, idx = component
        self.component = component[idx]
        self.bctype = get_enum(bctype, BoundaryType)
        self.value = list(val)
        self.options = list()

    def UseOption(self) -> None: ...

    def string(self):
        var, idx = self.component
        if idx is not None:
            var = f"{str(var)}.{idx}"
        string = join_fields(
            self.id, var, self.bctype, *self.value, *self.options, char="  "
        )
        return f"    {string}\n"


@dc.dataclass
class BoundaryCondition:
    patches: list[BCPatch] | None = None

    def AddPatch(self, *patch: BCPatch):
        if self.patches is None:
            self.patches = list()
        for p in patch:
            self.patches.append(p)

    def DefPatch(
        self,
        id: int,
        component: Variable,
        type: BoundaryType,
        *val: Expression | str | int | float,
    ):
        if self.patches is None:
            self.patches = list()
        self.patches.append(BCPatch(id, component, type, *val))

    def write(self, f: TextIO):
        if self.patches is None:
            f.write(f"  !Boundary-conditions-not-required\n\n")
        else:
            f.write(f"  !Boundary-patch-definitions\n")
            for p in self.patches:
                f.write(p.string())
            f.write("\n")


# Problems ----------------------------------------------------------------------


class _Problem(abc.ABC):

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def get_variables(self) -> dict[str, Variable]: ...

    @abc.abstractmethod
    def get_aux_vars(self) -> dict[str, Variable]: ...

    @abc.abstractmethod
    def get_bc_patches(self) -> list[BCPatch]: ...

    @abc.abstractmethod
    def write(self, f: TextIO) -> None: ...
