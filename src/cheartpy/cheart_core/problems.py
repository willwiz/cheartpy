from dataclasses import dataclass, field
from typing import Union, Literal
from .pytools import VoS
from .basetypes import *


@dataclass
class BCPatch:
    id: Union[int, str]
    component: str
    type: Literal[
        "dirichlet",
        "neumann",
        "neumann_ref",
        "neumann_nl",
        "stabilized_neumann",
        "consistent",
    ]
    value: Union[Expression, str, int]
    options: List[Union[str, int, float]] = field(default_factory=list)

    def string(self):
        return f'    {self.id}  {VoS(self.component)}  {self.type}  {VoS(self.value)}  {"  ".join(self.options)}\n'


@dataclass
class BoundaryCondition:
    patches: Optional[List[BCPatch]] = None

    def AddPatch(self, *patch: BCPatch):
        if self.patches is None:
            self.patches = list()
        for p in patch:
            self.patches.append(p)

    def DefPatch(
        self, id: Union[int, str], component: str, type: str, val: Union[str, int]
    ):
        self.patches.append(BCPatch(id, component, type, val))

    def write(self, f: TextIO):
        if self.patches is None:
            f.write(f"  !Boundary-conditions-not-required\n\n")
        else:
            f.write(f"  !Boundary-patch-definitions\n")
            for p in self.patches:
                f.write(p.string())
            f.write("\n")


# Matlaws -----------------------------------------------------------------------------
class Law:
    name: str
    aux_vars: Dict[str, Variable]


@dataclass
class Matlaw(Law):
    name: str
    parameters: List[float]
    aux_vars: Dict[str, Variable] = field(default_factory=dict)

    def string(self):
        return (
            f"  !ConstitutiveLaw={{{self.name}}}\n"
            f'    {"  ".join([str(i) for i in self.parameters])}\n'
        )


@dataclass
class FractionalVE(Law):
    alpha: float
    np: int
    Tf: float
    store: Union[Variable, str]
    Tscale: Optional[float] = 10.0
    name: str = "fractional-ve"
    InitPK2: Union[bool, int] = True
    ZeroPK2: bool = True
    Order: Literal[1, 2] = 2
    laws: List[Matlaw] = field(default_factory=list)
    aux_vars: Dict[str, Variable] = field(default_factory=dict)

    def __post_init__(self):
        self.aux_vars[self.store.name] = self.store

    def AddLaw(self, *law: Type[Law]):
        for v in law:
            self.laws.append(v)
            for k, x in v.aux_vars.items():
                self.aux_vars[k] = x

    def string(self):
        l = (
            f"  !ConstitutiveLaw={{{self.name}}}\n"
            f"    {VoS(self.store)}\n"
            f'    {self.alpha}  {self.np}  {self.Tf}  {"" if self.Tscale is None else self.Tscale}\n'
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


@dataclass
class FractionalDiffEQ(Law):
    alpha: float
    delta: float
    np: int
    Tf: float
    store: Union[Variable, str]
    Tscale: Optional[float] = 10.0
    name: str = "fractional-diffeq"
    InitPK2: Union[bool, int] = False
    ZeroPK2: bool = False
    Order: Literal[1, 2] = 2
    laws: List[Matlaw] = field(default_factory=list)
    aux_vars: Dict[str, Variable] = field(default_factory=dict)

    def __post_init__(self):
        self.aux_vars[self.store.name] = self.store

    def AddLaw(self, *law: Type[Law]):
        for v in law:
            self.laws.append(v)
            for k, x in v.aux_vars.items():
                self.aux_vars[k] = x

    def string(self):
        l = (
            f"  !ConstitutiveLaw={{{self.name}}}\n"
            f"    {VoS(self.store)}\n"
            f'    {self.alpha}  {self.np}  {self.Tf}  {self.delta}  {"" if self.Tscale is None else self.Tscale}\n'
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
                    + f"    frac{counter}  parm  {VoS(v.store)}  {v.alpha}  {v.np}  {v.Tf}  {sc}\n"
                )
                for law in v.laws:
                    l = (
                        l
                        + f'    frac{counter}  law   {law.name}  [{" ".join([str(i) for i in law.parameters])}]\n'
                    )
        return l


# Problems ----------------------------------------------------------------------


@dataclass
class Problem:
    name: str
    problem: str
    vars: Dict[str, Variable] = field(default_factory=dict)
    aux_vars: Dict[str, Variable] = field(default_factory=dict)
    options: Dict[str, List[str]] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    BC: BoundaryCondition = field(default_factory=BoundaryCondition)

    def UseVariable(self, req: str, var: Variable) -> None:
        self.vars[req] = var

    def UseOption(self, opt: str, *val: str) -> None:
        if val:
            self.options[opt] = list(val)
        else:
            self.flags.append(opt)

    def write(self, f: TextIO):
        f.write(f"!DefProblem={{{self.name}|{self.problem}}}\n")
        for k, v in self.vars.items():
            f.write(f"  !UseVariablePointer={{{k}|{v.name}}}\n")
        for k, v in self.options.items():
            f.write(f'  !{k}={{{"|".join([str(VoS(i)) for i in v])}}}\n')
        for v in self.flags:
            f.write(f"  !{v}\n")


@dataclass
class SolidProblem(Problem):
    problem: str = "quasi_static_elasticity"
    matlaws: List[Matlaw] = field(default_factory=list)

    def AddMatlaw(self, *law: Matlaw):
        for v in law:
            self.matlaws.append(v)
            for k, x in v.aux_vars.items():
                self.aux_vars[k] = x

    def write(self, f: TextIO):
        super().write(f)
        for v in self.matlaws:
            f.write(v.string())
        self.BC.write(f)


@dataclass
class L2Projection(Problem):
    problem: str = "l2solidprojection_problem"

    def UseVariable(self, req: Literal["Space", "Variable"], var: Variable) -> None:
        return super().UseVariable(req, var)

    def UseOption(
        self,
        opt: Literal[
            "Mechanical-Problem", "Projected-Variable", "Solid-Master-Override"
        ],
        *val: str,
    ) -> None:
        return super().UseOption(opt, *val)

    def write(self, f: TextIO):
        super().write(f)
        self.BC.write(f)


@dataclass
class NormProblem(Problem):
    problem: str = "norm_calculation"

    def UseVariable(
        self, req: Literal["Space", "Term1", "Term2"], var: Variable
    ) -> None:
        return super().UseVariable(req, var)

    def UseOption(
        self, opt: Literal["Boundary-normal", "Output-filename"], *val: str
    ) -> None:
        return super().UseOption(opt, *val)

    def write(self, f: TextIO):
        super().write(f)
        self.BC.write(f)
