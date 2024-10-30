import dataclasses as dc
from typing import Sequence, Union, TextIO, ValuesView

from cheartpy.cheart_core.interface.basis import _Expression
from ..pytools import get_enum, join_fields
from ..aliases import *
from ..interface import *


@dc.dataclass(init=False, slots=True)
class BCPatch(_BCPatch):
    id: Union[int, str]
    component: tuple[_Variable, int | None]
    bctype: BoundaryType
    values: list[_Expression | _Variable | str | int | float]
    options: list[str | int | float]

    def __init__(
        self,
        i: int,
        component: _Variable | tuple[_Variable, int | None],
        bctype: BOUNDARY_TYPE | BoundaryType,
        *val: _Expression | _Variable | str | int | float,
    ) -> None:
        self.id = i
        if isinstance(component, tuple):
            component, idx = component
        else:
            component, idx = component, None
        self.component = component[idx]
        self.bctype = get_enum(bctype, BoundaryType)
        self.values = list(val)
        self.options = list()

    def get_var_deps(self) -> ValuesView[_Variable]:
        vars = {str(v): v for v in self.values if isinstance(v, _Variable)}
        return vars.values()

    def get_expr_deps(self) -> ValuesView[_Expression]:
        exprs = {str(e): e for e in self.values if isinstance(e, _Expression)}
        return exprs.values()

    def UseOption(self) -> None: ...

    def string(self):
        var, idx = self.component
        if idx is not None:
            var = f"{str(var)}.{idx}"
        string = join_fields(
            self.id, var, self.bctype, *self.values, *self.options, char="  "
        )
        return f"    {string}\n"


@dc.dataclass(slots=True)
class BoundaryCondition(_BoundaryCondition):
    patches: list[_BCPatch] | None = None

    def get_vars_deps(self) -> ValuesView[_Variable]:
        if self.patches is None:
            return dict().values()
        vars = {str(v): v for patch in self.patches for v in patch.get_var_deps()}
        return vars.values()

    def get_expr_deps(self) -> ValuesView[_Expression]:
        if self.patches is None:
            return dict().values()
        exprs = {str(e): e for patch in self.patches for e in patch.get_expr_deps()}
        return exprs.values()

    def get_patches(self) -> list[_BCPatch] | None:
        return self.patches

    def AddPatch(self, *patch: _BCPatch):
        if self.patches is None:
            self.patches = list()
        for p in patch:
            self.patches.append(p)

    def DefPatch(
        self,
        id: int,
        component: _Variable,
        type: BoundaryType,
        *val: _Expression | str | int | float,
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
