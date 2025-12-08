import dataclasses as dc
from typing import TYPE_CHECKING, Literal

from cheartpy.fe.trait import IExpression, ILaw, IVariable

if TYPE_CHECKING:
    from collections.abc import Mapping, ValuesView

__all__ = ["FractionalDiffEQ", "FractionalVE", "Matlaw"]
# Matlaws -----------------------------------------------------------------------------


@dc.dataclass
class Matlaw(ILaw):
    name: str
    parameters: list[str | float | IExpression | IVariable] = dc.field(
        default_factory=list[str | float | IExpression | IVariable],
    )
    deps_var: dict[str, IVariable] = dc.field(default_factory=dict[str, IVariable])
    deps_expr: dict[str, IExpression] = dc.field(default_factory=dict[str, IExpression])

    def get_prob_vars(self) -> Mapping[str, IVariable]:
        return {str(s): s for s in self.parameters if isinstance(s, IVariable)}

    def add_var_deps(self, *var: IVariable) -> None:
        for v in var:
            if str(v) not in self.deps_var:
                self.deps_var[str(v)] = v

    def add_expr_deps(self, *exprs: IExpression) -> None:
        for e in exprs:
            if str(e) not in self.deps_expr:
                self.deps_expr[str(e)] = e

    def get_var_deps(self) -> ValuesView[IVariable]:
        return {**self.get_prob_vars(), **self.deps_var}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _p_exprs_ = {str(s): s for s in self.parameters if isinstance(s, IExpression)}
        return {**_p_exprs_, **self.deps_expr}.values()

    def string(self) -> str:
        return (
            f"  !ConstitutiveLaw={{{self.name}}}\n"
            f"    {'  '.join([str(i) for i in self.parameters])}\n"
        )


@dc.dataclass
class FractionalVE(ILaw):
    alpha: float
    np: int
    Tf: float
    store: IVariable
    Tscale: float | None = 10.0
    name: Literal["fractional-ve"] = "fractional-ve"
    InitPK2: bool = True
    ZeroPK2: bool = True
    Order: Literal[1, 2] = 2
    laws: list[Matlaw] = dc.field(default_factory=list[Matlaw])
    deps_var: dict[str, IVariable] = dc.field(default_factory=dict[str, IVariable])
    deps_expr: dict[str, IExpression] = dc.field(default_factory=dict[str, IExpression])

    def get_prob_vars(self) -> Mapping[str, IVariable]:
        return {}

    def add_var_deps(self, *var: IVariable) -> None:
        for v in var:
            if str(v) not in self.deps_var:
                self.deps_var[str(v)] = v

    def add_expr_deps(self, *expr: IExpression) -> None:
        for e in expr:
            if str(e) not in self.deps_expr:
                self.deps_expr[str(e)] = e

    def get_var_deps(self) -> ValuesView[IVariable]:
        _vars_ = {str(self.store): self.store}
        for d in self.laws:
            _vars_.update({str(v): v for v in d.get_var_deps()})
            _vars_.update(d.get_prob_vars())
        return {**_vars_, **self.deps_var}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for d in self.laws for e in d.get_expr_deps()}
        return {**_expr_, **self.deps_expr}.values()

    def __post_init__(self) -> None:
        self.deps_var[str(self.store)] = self.store

    def add_law(self, *laws: Matlaw) -> None:
        for v in laws:
            self.laws.append(v)
            for k, x in v.deps_var.items():
                self.deps_var[k] = x

    def string(self) -> str:
        s = (
            f"  !ConstitutiveLaw={{{self.name}}}\n"
            f"    {self.store!s}\n"
            f"    {self.alpha}  {self.np}  {self.Tf}  {
                '' if self.Tscale is None else self.Tscale
            }\n"
        )
        if self.InitPK2:
            s = s + f"    InitPK2  {self.InitPK2 if (type(self.InitPK2) is int) else ''}\n"
        if self.ZeroPK2:
            s = s + "    ZeroPK2\n"
        if self.Order == 1:
            s = s + f"    Order {self.Order}\n"
        for v in self.laws:
            s = s + f"    {v.name}  [{' '.join([str(i) for i in v.parameters])}]\n"
        return s


@dc.dataclass
class FractionalDiffEQ(ILaw):
    alpha: float
    delta: float
    np: int
    Tf: float
    store: IVariable
    Tscale: float | None = 10.0
    name: str = "fractional-diffeq"
    InitPK2: bool = False
    ZeroPK2: bool = False
    Order: Literal[1, 2] = 2
    laws: list[Matlaw | FractionalVE] = dc.field(default_factory=list[Matlaw | FractionalVE])
    deps_var: dict[str, IVariable] = dc.field(default_factory=dict[str, IVariable])
    deps_expr: dict[str, IExpression] = dc.field(default_factory=dict[str, IExpression])

    def get_prob_vars(self) -> Mapping[str, IVariable]:
        return {}

    def add_var_deps(self, *var: IVariable) -> None:
        for v in var:
            if str(v) not in self.deps_var:
                self.deps_var[str(v)] = v

    def add_expr_deps(self, *expr: IExpression) -> None:
        for e in expr:
            if str(e) not in self.deps_expr:
                self.deps_expr[str(e)] = e

    def get_var_deps(self) -> ValuesView[IVariable]:
        _vars_ = {str(self.store): self.store}
        for d in self.laws:
            _vars_.update({str(v): v for v in d.get_var_deps()})
            _vars_.update(d.get_prob_vars())
        return {**_vars_, **self.deps_var}.values()

    def get_expr_deps(self) -> ValuesView[IExpression]:
        _expr_ = {str(e): e for d in self.laws for e in d.get_expr_deps()}
        return {**_expr_, **self.deps_expr}.values()

    def __post_init__(self) -> None:
        self.deps_var[str(self.store)] = self.store
        for v in self.laws:
            for k, x in v.deps_var.items():
                self.deps_var[k] = x

    def add_law(self, *law: Matlaw | FractionalVE) -> None:
        for v in law:
            self.laws.append(v)
            for k, x in v.deps_var.items():
                self.deps_var[k] = x

    def string(self) -> str:
        s = (
            f"  !ConstitutiveLaw={{{self.name}}}\n"
            f"    {self.store!s}\n"
            f"    {self.alpha}  {self.np}  {self.Tf}  {self.delta}  {
                '' if self.Tscale is None else self.Tscale
            }\n"
        )
        if self.InitPK2:
            s = s + f"    InitPK2  {self.InitPK2 if (type(self.InitPK2) is int) else ''}\n"
        if self.ZeroPK2:
            s = s + "    ZeroPK2\n"
        if self.Order == 1:
            s = s + f"    Order {self.Order}\n"
        counter = 0
        for v in self.laws:
            if isinstance(v, FractionalVE):
                counter = counter + 1
                sc = "" if v.Tscale is None else v.Tscale
                s = s + f"    frac{counter}  parm  {v.store!s}  {v.alpha}  {v.np}  {v.Tf}  {sc}\n"
                for law in v.laws:
                    s = (
                        s
                        + f"    frac{counter}  law   {law.name}  "
                        + f"[{' '.join([str(i) for i in law.parameters])}]\n"
                    )
            else:
                s = s + f"    HE  law  {v.name}  [{' '.join([str(i) for i in v.parameters])}]\n"
        return s
