from __future__ import annotations

__all__ = ["XMLWriters", "XMLElement"]
from typing import Any, Callable, TextIO
from ..var_types import *

WRITERSIGS = (
    Callable[[TextIO, int, int], None]
    | Callable[[TextIO, float, int], None]
    | Callable[[TextIO, Vec[f64], int], None]
    | Callable[[TextIO, Vec[int_t], int], None]
)


class XMLWriters:
    @staticmethod
    def PointWriter(fout: TextIO, point: Vec[f64], level: int = 0) -> None:
        fout.write(f'{" "*level}{point[0]: .16f} {point[1]: .16f} {point[2]: .16f}\n')

    @staticmethod
    def IntegerWriter(fout: TextIO, id: int, level: int = 0) -> None:
        fout.write(f'{" "*level}{id:d}\n')

    @staticmethod
    def FloatArrWriter(fout: TextIO, arr: Vec[f64], level: int = 0) -> None:
        fout.write(" " * (level - 1))
        for p in arr:
            fout.write(f" {p:< .16f}")
        fout.write("\n")


class XMLElement:
    data: Mat[Any] | None
    datawriter: WRITERSIGS
    attribs: str
    subelems: list[XMLElement]

    def __init__(self, tag: str, **attribs: str) -> None:
        self.tag = tag
        string = ""
        for k, v in attribs.items():
            string = string + f' {k}="{v}"'
        self.attribs = string
        self.data = None
        self.datawriter = XMLWriters.PointWriter
        self.subelems = list()

    def add_elem(self, elem: XMLElement):
        self.subelems.append(elem)
        return elem

    def add_data(
        self,
        arr: Arr[Any, Any],
        writer: WRITERSIGS = XMLWriters.PointWriter,
    ) -> None:
        self.data = arr
        self.datawriter = writer

    def write(self, fout: TextIO, level: int = 0) -> None:
        fout.write(f'{" "*level}<{self.tag}{self.attribs}>\n')
        for elem in self.subelems:
            elem.write(fout, level + 2)
        if self.data is not None:
            for d in self.data:
                self.datawriter(fout, d, level + 2)
        fout.write(f'{" "*level}</{self.tag}>\n')
