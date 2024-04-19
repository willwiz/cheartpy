import dataclasses as dc
from typing import TextIO
from ..pytools import hline


@dc.dataclass
class TimeScheme:
    name: str
    start: int
    stop: int
    value: float | str  # step or file

    def write(self, f: TextIO):
        f.write(hline("Define Time Scheme"))
        f.write(f"!DefTimeStepScheme={{{self.name}}}\n")
        f.write(f"  {self.start}  {self.stop}  {self.value}\n")
