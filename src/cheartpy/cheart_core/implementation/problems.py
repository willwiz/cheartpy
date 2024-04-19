import dataclasses as dc
from typing import Union, TextIO

from cheartpy.cheart_core.interface.basis import _Expression, _Variable
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
        *val: _Expression | str | int | float,
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

    def get_values(self) -> list[_Expression | _Variable | str | int | float]:
        return self.values

    def UseOption(self) -> None: ...

    def string(self):
        var, idx = self.component
        if idx is not None:
            var = f"{str(var)}.{idx}"
        string = join_fields(
            self.id, var, self.bctype, *self.values, *self.options, char="  "
        )
        return f"    {string}\n"


@dc.dataclass
class BoundaryCondition:
    patches: list[_BCPatch] | None = None

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
