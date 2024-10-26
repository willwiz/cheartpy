import abc
import dataclasses as dc
from typing import Literal
from ...interface import _Variable, _Expression

# Matlaws -----------------------------------------------------------------------------


class Law(abc.ABC):

    @abc.abstractmethod
    def string(self) -> str: ...

    @abc.abstractmethod
    def get_aux_vars(self) -> dict[str, _Variable]: ...


@dc.dataclass
class Matlaw(Law):
    name: str
    parameters: list[str | float | _Expression] = dc.field(default_factory=list)
    aux_vars: dict[str, _Variable] = dc.field(default_factory=dict)

    def get_aux_vars(self) -> dict[str, _Variable]:
        return self.aux_vars

    def string(self):
        return (
            f"  !ConstitutiveLaw={{{self.name}}}\n"
            f'    {"  ".join([str(i) for i in self.parameters])}\n'
        )


@dc.dataclass
class FractionalVE(Law):
    alpha: float
    np: int
    Tf: float
    store: _Variable
    Tscale: float | None = 10.0
    name: Literal["fractional-ve"] = "fractional-ve"
    InitPK2: bool = True
    ZeroPK2: bool = True
    Order: Literal[1, 2] = 2
    laws: list[Matlaw] = dc.field(default_factory=list)
    aux_vars: dict[str, _Variable] = dc.field(default_factory=dict)

    def get_aux_vars(self) -> dict[str, _Variable]:
        return self.aux_vars

    def __post_init__(self):
        self.aux_vars[str(self.store)] = self.store

    def AddLaw(self, *laws: Matlaw):
        for v in laws:
            self.laws.append(v)
            for k, x in v.aux_vars.items():
                self.aux_vars[k] = x

    def string(self):
        l = (
            f"  !ConstitutiveLaw={{{self.name}}}\n"
            f"    {str(self.store)}\n"
            f'    {self.alpha}  {self.np}  {self.Tf}  {
                "" if self.Tscale is None else self.Tscale}\n'
        )
        if self.InitPK2:
            l = (
                l
                + f'    InitPK2  {self.InitPK2 if (type(self.InitPK2) is int) else ""}\n'
            )
        if self.ZeroPK2:
            l = l + f"    ZeroPK2\n"
        if self.Order != 2:
            l = l + f"    Order 1\n"
        for v in self.laws:
            l = l + f'    {v.name}  [{" ".join([str(i) for i in v.parameters])}]\n'
        return l


@dc.dataclass
class FractionalDiffEQ(Law):
    alpha: float
    delta: float
    np: int
    Tf: float
    store: _Variable
    Tscale: float | None = 10.0
    name: str = "fractional-diffeq"
    InitPK2: bool = False
    ZeroPK2: bool = False
    Order: Literal[1, 2] = 2
    laws: list[Matlaw | FractionalVE] = dc.field(default_factory=list)
    aux_vars: dict[str, _Variable] = dc.field(default_factory=dict)

    def get_aux_vars(self) -> dict[str, _Variable]:
        return self.aux_vars

    def __post_init__(self):
        self.aux_vars[str(self.store)] = self.store
        for v in self.laws:
            for k, x in v.aux_vars.items():
                self.aux_vars[k] = x

    def AddLaw(self, *law: Matlaw | FractionalVE):
        for v in law:
            self.laws.append(v)
            for k, x in v.aux_vars.items():
                self.aux_vars[k] = x

    def string(self):
        l = (
            f"  !ConstitutiveLaw={{{self.name}}}\n"
            f"    {str(self.store)}\n"
            f'    {self.alpha}  {self.np}  {self.Tf}  {self.delta}  {
                "" if self.Tscale is None else self.Tscale}\n'
        )
        if self.InitPK2:
            l = (
                l
                + f'    InitPK2  {self.InitPK2 if (type(self.InitPK2) is int) else ""}\n'
            )
        if self.ZeroPK2:
            l = l + "    ZeroPK2\n"
        if self.Order != 2:
            l = l + "    Order 1\n"
        counter = 0
        for v in self.laws:
            if isinstance(v, Matlaw):
                l = (
                    l
                    + f'    HE  law  {v.name}  [{" ".join([str(i) for i in v.parameters])}]\n'
                )
            elif isinstance(v, FractionalVE):
                counter = counter + 1
                sc = "" if v.Tscale is None else v.Tscale
                l = (
                    l
                    + f"    frac{counter}  parm  {str(v.store)}  {v.alpha}  {v.np}  {v.Tf}  {sc}\n"
                )
                for law in v.laws:
                    l = (
                        l
                        + f'    frac{counter}  law   {law.name}  [{" ".join([str(i) for i in law.parameters])}]\n'
                    )
        return l
