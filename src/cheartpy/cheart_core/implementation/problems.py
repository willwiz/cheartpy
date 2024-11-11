__all__ = ["BCPatch", "BoundaryCondition"]
import dataclasses as dc
from typing import TextIO, ValuesView
from ..pytools import get_enum, join_fields
from ..aliases import *
from ..interface import *


@dc.dataclass(init=False, slots=True)
class BCPatch(IBCPatch):
    id: int | str
    component: tuple[IVariable, int | None]
    bctype: BoundaryType
    values: list[IExpression | IVariable | str | int | float]
    options: list[str | int | float]

    def __hash__(self) -> int:
        return hash(
            (
                self.id,
                str(self.component[0]),
                str(self.component[1]),
                self.bctype.value,
                *[str(v) for v in self.values],
                *self.options,
            )
        )

    def __init__(
        self,
        i: int,
        component: IVariable | tuple[IVariable, int | None],
        bctype: BOUNDARY_TYPE | BoundaryType,
        *val: IExpression | IVariable | str | int | float,
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

    def get_var_deps(self) -> ValuesView[IVariable]:
        vars = {str(v): v for v in self.values if isinstance(v, IVariable)}
        return vars.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        exprs = {str(e): e for e in self.values if isinstance(e, IExpression)}
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


class BoundaryCondition(IBoundaryCondition):
    __slots__ = ["patches"]
    patches: dict[int, IBCPatch] | None

    def __init__(self, patch: list[IBCPatch] | None = None) -> None:
        if patch is None:
            self.patches = None
            return
        self.patches = dict()
        for p in patch:
            self.patches[hash(p)] = p

    def get_vars_deps(self):
        if self.patches is None:
            false_dict: dict[int, IVariable] = dict()
            return false_dict.values()
        vars = {
            str(v): v for patch in self.patches.values() for v in patch.get_var_deps()
        }
        return vars.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        if self.patches is None:
            false_dict: dict[int, IExpression] = dict()
            return false_dict.values()
        exprs = {
            str(e): e for patch in self.patches.values() for e in patch.get_expr_deps()
        }
        return exprs.values()

    def get_patches(self) -> ValuesView[IBCPatch] | None:
        if self.patches is None:
            return None
        return self.patches.values()

    def AddPatch(self, *patch: IBCPatch):
        if self.patches is None:
            self.patches = dict()
        for p in patch:
            self.patches[hash(p)] = p

    def write(self, f: TextIO):
        if self.patches is None:
            f.write(f"  !Boundary-conditions-not-required\n\n")
        else:
            f.write(f"  !Boundary-patch-definitions\n")
            for p in self.patches.values():
                f.write(p.string())
            f.write("\n")


# Problems ----------------------------------------------------------------------
