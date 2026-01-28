import dataclasses as dc
from typing import TYPE_CHECKING, TextIO

from cheartpy.fe.aliases import BoundaryEnum, BoundaryType
from cheartpy.fe.string_tools import get_enum, join_fields
from cheartpy.fe.trait import BC_VALUE, IBCPatch, IBoundaryCondition, IExpression, IVariable

if TYPE_CHECKING:
    from collections.abc import Sequence, ValuesView

__all__ = ["BCPatch", "BoundaryCondition"]


@dc.dataclass(init=False, slots=True)
class BCPatch(IBCPatch):
    id: int | str
    component: tuple[IVariable, int | None]
    bctype: BoundaryEnum
    values: list[BC_VALUE]
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
            ),
        )

    def __init__(
        self,
        i: int,
        component: IVariable | tuple[IVariable, int | None],
        bctype: BoundaryType | BoundaryEnum,
        *val: BC_VALUE,
    ) -> None:
        self.id = i
        if isinstance(component, tuple):
            component, idx = component
        else:
            idx = None
        self.component = component[idx]
        self.bctype = get_enum(bctype, BoundaryEnum)
        self.values = list(val)
        self.options = []

    def get_var_deps(self) -> ValuesView[IVariable]:
        variables = {str(v): v for v in self.values if isinstance(v, IVariable)}
        for v in self.values:
            if isinstance(v, tuple):
                variables[str(v[0])] = v[0]
        return variables.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        exprs = {str(e): e for e in self.values if isinstance(e, IExpression)}
        return exprs.values()

    def use_option(self) -> None: ...

    def string(self) -> str:
        var, idx = self.component
        if idx is not None:
            var = f"{var!s}.{idx}"
        string = join_fields(
            self.id,
            var,
            self.bctype,
            *self.values,
            *self.options,
            char="  ",
        )
        return f"    {string}\n"


class BoundaryCondition(IBoundaryCondition):
    __slots__ = ["patches"]
    patches: dict[int, IBCPatch] | None

    def __init__(self, patch: Sequence[IBCPatch] | None = None) -> None:
        if patch is None:
            self.patches = None
            return
        self.patches = {}
        for p in patch:
            p_hash = hash(p)
            if p_hash not in self.patches:
                self.patches[p_hash] = p

    def get_vars_deps(self) -> ValuesView[IVariable]:
        if self.patches is None:
            false_dict: dict[int, IVariable] = {}
            return false_dict.values()
        variables = {str(v): v for patch in self.patches.values() for v in patch.get_var_deps()}
        return variables.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        if self.patches is None:
            false_dict: dict[int, IExpression] = {}
            return false_dict.values()
        exprs = {str(e): e for patch in self.patches.values() for e in patch.get_expr_deps()}
        return exprs.values()

    def get_patches(self) -> ValuesView[IBCPatch] | None:
        if self.patches is None:
            return None
        return self.patches.values()

    def add_patch(self, *patch: IBCPatch) -> None:
        if self.patches is None:
            self.patches = {}
        for p in patch:
            self.patches[hash(p)] = p

    def write(self, f: TextIO) -> None:
        if self.patches is None or len(self.patches) == 0:
            f.write("  !Boundary-conditions-not-required\n\n")
        else:
            f.write("  !Boundary-patch-definitions\n")
            f.writelines(p.string() for p in self.patches.values())
            f.write("\n")


# Problems ----------------------------------------------------------------------
