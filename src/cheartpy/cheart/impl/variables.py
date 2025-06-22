from __future__ import annotations

__all__ = ["Variable"]
import dataclasses as dc
from typing import TYPE_CHECKING, Literal, Self, TextIO, overload

from cheartpy.cheart.aliases import (
    VARIABLE_UPDATE_SETTING,
    VariableExportFormat,
    VariableUpdateSetting,
)
from cheartpy.cheart.pytools import get_enum, join_fields
from cheartpy.cheart.trait import ICheartTopology, IExpression, IVariable

if TYPE_CHECKING:
    from collections.abc import ValuesView


@dc.dataclass(slots=True)
class Variable(IVariable):
    name: str
    topology: ICheartTopology
    dim: int
    data: str | None = None
    fmt: VariableExportFormat = VariableExportFormat.TXT
    freq: int = 1
    loop_step: int | None = None
    setting: tuple[VariableUpdateSetting, str | IExpression] | None = None
    deps_expr: dict[str, IExpression] = dc.field(default_factory=dict[str, IExpression])

    def __repr__(self) -> str:
        return self.name

    def __getitem__[T: int | None](self, key: T) -> tuple[Self, T]:
        return (self, key)

    def __bool__(self) -> Literal[True]:
        return True

    @property
    def order(self) -> Literal[0, 1, 2, None]:
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
        val: str,
    ) -> None: ...

    def add_setting(
        self,
        task: VARIABLE_UPDATE_SETTING,
        val: str | IExpression,
    ):
        match task, val:
            case "INIT_EXPR" | "TEMPORAL_UPDATE_EXPR", IExpression():
                self.setting = (get_enum(task, VariableUpdateSetting), val)
                self.deps_expr[str(val)] = val
            case "TEMPORAL_UPDATE_FILE", str():
                self.setting = (get_enum(task, VariableUpdateSetting), val)
            case "TEMPORAL_UPDATE_FILE_LOOP", str():
                self.setting = (get_enum(task, VariableUpdateSetting), val)
                if self.loop_step is None:
                    self.loop_step = self.freq
            case _:
                msg = (f"Setting for variable {self.name} does not match correct type",)
                raise ValueError(msg)

    def set_format(self, fmt: Literal["TXT", "BINARY", "MMAP"]) -> None:
        self.fmt = VariableExportFormat[fmt]

    def add_data(self, data: str | None) -> None:
        self.data = data

    def get_data(self) -> str | None:
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
        string = join_fields(
            self.name,
            self.topology if self.topology else "null_topology",
            self.data,
            self.dim,
        )
        f.write(f"!DefVariablePointer={{{string}}}\n")
        if self.fmt == VariableExportFormat.BINARY:
            f.write(f"  !SetVariablePointer={{{self.name}|ReadBinary}}\n")
        elif self.fmt == VariableExportFormat.MMAP:
            f.write(f"  !SetVariablePointer={{{self.name}|ReadMMap}}\n")
        if self.setting:
            string = join_fields(self.name, self.setting[0], self.setting[1])
            f.write(f"  !SetVariablePointer={{{string}}}\n")
