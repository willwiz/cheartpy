import dataclasses as dc
from typing import TYPE_CHECKING, TextIO

from cheartpy.fe.string_tools import hline
from cheartpy.fe.trait import ITimeScheme

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["TimeScheme"]


@dc.dataclass(slots=True)
class TimeScheme(ITimeScheme):
    name: str
    start: int
    stop: int
    value: float | str | Path  # step or file

    def __repr__(self) -> str:
        return self.name

    def write(self, f: TextIO) -> None:
        f.write(hline("Define Time Scheme"))
        f.write(f"!DefTimeStepScheme={{{self.name}}}\n")
        f.write(f"  {self.start}  {self.stop}  {self.value}\n")
