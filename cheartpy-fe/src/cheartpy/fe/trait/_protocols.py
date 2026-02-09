from typing import Protocol, TextIO


class HasWriter(Protocol):
    def write(self, f: TextIO) -> None: ...
