import dataclasses as dc
from typing import Literal, Mapping, ValuesView

from cheartpy.cheart_core.interface.basis import _Expression, _Variable
from ...interface import *

# Matlaws -----------------------------------------------------------------------------


@dc.dataclass
class Matlaw(_Law):
    name: str
    parameters: list[str | float | _Expression | _Variable] = dc.field(
        default_factory=list
    )
    deps_var: dict[str, _Variable] = dc.field(default_factory=dict)
    deps_expr: dict[str, _Expression] = dc.field(default_factory=dict)

    def get_prob_vars(self) -> Mapping[str, _Variable]:
        _p_vars_ = {str(s): s for s in self.parameters if isinstance(s, _Variable)}
        return _p_vars_

    def add_var_deps(self, *vars: _Variable) -> None:
        for v in vars:
            if str(v) not in self.deps_var:
                self.deps_var[str(v)] = v

    def add_expr_deps(self, *exprs: _Expression) -> None:
        for e in exprs:
            if str(e) not in self.deps_expr:
                self.deps_expr[str(e)] = e

    def get_var_deps(self) -> ValuesView[_Variable]:
        return {**self.get_prob_vars(), **self.deps_var}.values()

    def get_expr_deps(self) -> ValuesView[_Expression]:
        _p_exprs_ = {str(s): s for s in self.parameters if isinstance(s, _Expression)}
        return {**_p_exprs_, **self.deps_expr}.values()

    def string(self):
        return (
            f"  !ConstitutiveLaw={{{self.name}}}\n"
            f'    {"  ".join([str(i) for i in self.parameters])}\n'
        )


@dc.dataclass
class FractionalVE(_Law):
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
    deps_var: dict[str, _Variable] = dc.field(default_factory=dict)
    deps_expr: dict[str, _Expression] = dc.field(default_factory=dict)

    def get_prob_vars(self) -> Mapping[str, _Variable]:
        return dict()

    def add_var_deps(self, *vars: _Variable) -> None:
        for v in vars:
            if str(v) not in self.deps_var:
                self.deps_var[str(v)] = v

    def add_expr_deps(self, *exprs: _Expression) -> None:
        for e in exprs:
            if str(e) not in self.deps_expr:
                self.deps_expr[str(e)] = e

    def get_var_deps(self) -> ValuesView[_Variable]:
        _vars_ = {str(self.store): self.store}
        for d in self.laws:
            _vars_.update({str(v): v for v in d.get_var_deps()})
            _vars_.update(d.get_prob_vars())
        return {**_vars_, **self.deps_var}.values()

    def get_expr_deps(self) -> ValuesView[_Expression]:
        _expr_ = {str(e): e for d in self.laws for e in d.get_expr_deps()}
        return {**_expr_, **self.deps_expr}.values()

    def __post_init__(self):
        self.deps_var[str(self.store)] = self.store

    def AddLaw(self, *laws: Matlaw):
        for v in laws:
            self.laws.append(v)
            for k, x in v.deps_var.items():
                self.deps_var[k] = x

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
class FractionalDiffEQ(_Law):
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
    deps_var: dict[str, _Variable] = dc.field(default_factory=dict)
    deps_expr: dict[str, _Expression] = dc.field(default_factory=dict)

    def get_prob_vars(self) -> Mapping[str, _Variable]:
        return dict()

    def add_var_deps(self, *vars: _Variable) -> None:
        for v in vars:
            if str(v) not in self.deps_var:
                self.deps_var[str(v)] = v

    def add_expr_deps(self, *exprs: _Expression) -> None:
        for e in exprs:
            if str(e) not in self.deps_expr:
                self.deps_expr[str(e)] = e

    def get_var_deps(self) -> ValuesView[_Variable]:
        _vars_ = {str(self.store): self.store}
        for d in self.laws:
            _vars_.update({str(v): v for v in d.get_var_deps()})
            _vars_.update(d.get_prob_vars())
        return {**_vars_, **self.deps_var}.values()

    def get_expr_deps(self) -> ValuesView[_Expression]:
        _expr_ = {str(e): e for d in self.laws for e in d.get_expr_deps()}
        return {**_expr_, **self.deps_expr}.values()

    def __post_init__(self):
        self.deps_var[str(self.store)] = self.store
        for v in self.laws:
            for k, x in v.deps_var.items():
                self.deps_var[k] = x

    def AddLaw(self, *law: Matlaw | FractionalVE):
        for v in law:
            self.laws.append(v)
            for k, x in v.deps_var.items():
                self.deps_var[k] = x

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
