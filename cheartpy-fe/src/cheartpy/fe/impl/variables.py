import dataclasses as dc
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self, TextIO, overload

from cheartpy.fe.aliases import (
    VariableExportEnum,
    VariableUpdateEnum,
    VariableUpdateSetting,
)
from cheartpy.fe.trait import ICheartTopology, IExpression, IVariable
from cheartpy.fe.utils import get_enum, join_fields

if TYPE_CHECKING:
    from collections.abc import ValuesView

__all__ = ["Variable"]


@dc.dataclass(slots=True)
class Variable(IVariable):
    name: str
    topology: ICheartTopology
    dim: int
    data: Path | None = None
    fmt: VariableExportEnum = VariableExportEnum.TXT
    freq: int = 1
    loop_step: int | None = None
    setting: tuple[VariableUpdateEnum, Path | str | IExpression] | None = None
    deps_expr: dict[str, IExpression] = dc.field(default_factory=dict[str, IExpression])

    def __repr__(self) -> str:
        return self.name

    def __getitem__[T: int | None](self, key: T) -> tuple[Self, T]:
        return (self, key)

    def __bool__(self) -> Literal[True]:
        return True

    @property
    def order(self) -> Literal[0, 1, 2] | None:
        return self.topology.order

    def idx(self, key: int) -> str:
        return f"{self.name}.{key}"

    def get_dim(self) -> int:
        return self.dim

    @overload
    def add_setting(
        self,
        task: Literal["INIT_EXPR", "TEMPORAL_UPDATE_EXPR"],
        val: IExpression,
    ) -> None: ...

    @overload
    def add_setting(
        self,
        task: Literal["TEMPORAL_UPDATE_FILE", "TEMPORAL_UPDATE_FILE_LOOP"],
        val: Path | str,
    ) -> None: ...

    def add_setting(
        self,
        task: VariableUpdateSetting,
        val: Path | str | IExpression,
    ):
        match task, val:
            case "INIT_EXPR" | "TEMPORAL_UPDATE_EXPR", IExpression():
                self.setting = (get_enum(task, VariableUpdateEnum), val)
                self.deps_expr[str(val)] = val
            case "TEMPORAL_UPDATE_FILE", str() | Path():
                self.setting = (get_enum(task, VariableUpdateEnum), val)
            case "TEMPORAL_UPDATE_FILE_LOOP", str() | Path():
                self.setting = (get_enum(task, VariableUpdateEnum), val)
                if self.loop_step is None:
                    self.loop_step = self.freq
            case _:
                msg = (f"Setting for variable {self.name} does not match correct type",)
                raise ValueError(msg)

    def set_format(self, fmt: Literal["TXT", "BINARY", "MMAP"]) -> None:
        self.fmt = VariableExportEnum[fmt]

    def add_data(self, data: Path | str | None) -> None:
        self.data = Path(data) if data is not None else None

    def get_data(self) -> Path | None:
        return self.data

    def get_top(self) -> ICheartTopology:
        return self.topology

    def get_expr_deps(
        self,
    ) -> ValuesView[IExpression]:
        return self.deps_expr.values()

    def set_export_frequency(self, v: int) -> None:
        self.freq = v

    def get_export_frequency(self) -> int:
        return self.freq

    def write(self, f: TextIO) -> None:
        string = join_fields(self.name, self.topology or "null_topology", self.data, self.dim)
        f.write(f"!DefVariablePointer={{{string}}}\n")
        if self.fmt == VariableExportEnum.BINARY:
            f.write(f"  !SetVariablePointer={{{self.name}|ReadBinary}}\n")
        elif self.fmt == VariableExportEnum.MMAP:
            f.write(f"  !SetVariablePointer={{{self.name}|ReadMMap}}\n")
        if self.setting:
            string = join_fields(self.name, self.setting[0], self.setting[1])
            f.write(f"  !SetVariablePointer={{{string}}}\n")
