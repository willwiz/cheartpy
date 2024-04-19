import dataclasses as dc
from typing import TextIO
from ..pytools import hline
from ..interface import *


@dc.dataclass(slots=True)
class TimeScheme(_TimeScheme):
    name: str
    start: int
    stop: int
    value: float | str  # step or file

    def __repr__(self) -> str:
        return self.name

    def write(self, f: TextIO):
        f.write(hline("Define Time Scheme"))
        f.write(f"!DefTimeStepScheme={{{self.name}}}\n")
        f.write(f"  {self.start}  {self.stop}  {self.value}\n")
