import dataclasses as dc
from typing import Literal, TextIO
from ..variables import Variable
from ..problems import BoundaryCondition, Problem


@dc.dataclass
class NormProblem(Problem):
    problem: str = "norm_calculation"
    bc: BoundaryCondition = BoundaryCondition()

    def UseVariable(
        self, req: Literal["Space", "Term1", "Term2"], var: Variable
    ) -> None:
        ...

    def UseOption(
        self, opt: Literal["Boundary-normal", "Output-filename"], *val: str
    ) -> None:
        ...

    def write(self, f: TextIO):
        self.bc.write(f)
