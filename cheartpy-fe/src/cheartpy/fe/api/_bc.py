from typing import TYPE_CHECKING

from cheartpy.fe.aliases import BoundaryEnum, BoundaryType
from cheartpy.fe.impl import BCPatch, BoundaryCondition
from cheartpy.fe.utils import get_enum

if TYPE_CHECKING:
    from cheartpy.fe.trait import BC_VALUE, IBCPatch, IBoundaryCondition, IVariable


def create_bcpatch(
    label: int,
    var: IVariable | tuple[IVariable, int | None],
    kind: BoundaryType,
    *val: BC_VALUE,
) -> IBCPatch:
    _kind = get_enum(kind, BoundaryEnum)
    return BCPatch(label, var, _kind, *val)


def create_bc(*val: IBCPatch) -> IBoundaryCondition:
    if len(val) > 0:
        return BoundaryCondition(val)
    return BoundaryCondition()
