import dataclasses as dc
from typing import Literal, TextIO
from ..variables import Variable
from ..problems import BoundaryCondition, Problem


@dc.dataclass
class L2Projection(Problem):
    problem: str = "l2solidprojection_problem"
    bc: BoundaryCondition = BoundaryCondition()

    def UseVariable(self, req: Literal["Space", "Variable"], var: Variable) -> None:
        ...

    def UseOption(
        self,
        opt: Literal[
            "Mechanical-Problem", "Projected-Variable", "Solid-Master-Override"
        ],
        *val: str,
    ) -> None:
        ...

    def write(self, f: TextIO):
        self.bc.write(f)
