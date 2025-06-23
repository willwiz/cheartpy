from __future__ import annotations

__all__ = ["TimeScheme"]
import dataclasses as dc
from typing import TextIO

from cheartpy.cheart.string_tools import hline
from cheartpy.cheart.trait import ITimeScheme


@dc.dataclass(slots=True)
class TimeScheme(ITimeScheme):
    name: str
    start: int
    stop: int
    value: float | str  # step or file

    def __repr__(self) -> str:
        return self.name

    def write(self, f: TextIO) -> None:
        f.write(hline("Define Time Scheme"))
        f.write(f"!DefTimeStepScheme={{{self.name}}}\n")
        f.write(f"  {self.start}  {self.stop}  {self.value}\n")
