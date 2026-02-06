from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, Unpack

from cheartpy.fe.aliases import VariableExportEnum, VariableExportFormat
from cheartpy.fe.impl import NullTopology, Variable
from cheartpy.fe.string_tools import get_enum

if TYPE_CHECKING:
    from cheartpy.fe.trait import ICheartTopology, IVariable


class _ExtraCreateVarOptions(TypedDict, total=False):
    fmt: VariableExportFormat
    freq: int
    loop_step: int | None


def create_variable(
    name: str,
    top: ICheartTopology | None,
    dim: int = 3,
    data: Path | str | None = None,
    **kwargs: Unpack[_ExtraCreateVarOptions],
) -> IVariable:
    fmt = get_enum(kwargs.get("fmt", VariableExportEnum.TXT), VariableExportEnum)
    top = NullTopology() if top is None else top
    data = Path(data) if data is not None else None
    return Variable(name, top, dim, data, fmt, kwargs.get("freq", 1), kwargs.get("loop_step"))
