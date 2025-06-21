from __future__ import annotations

import numpy as np

__all__ = ["XMLElement", "XMLWriters"]
from collections.abc import Callable
from typing import TextIO

# from ..var_types import *
from arraystubs import Arr1, Arr2

type WRITERSIGS = Callable[[TextIO, Arr1[np.generic], int], None]


class XMLWriters:
    @staticmethod
    def int_writer(fout: TextIO, idx: Arr1[np.integer], level: int) -> None:
        fout.write(" " * level)
        fout.writelines(f"{p:d} " for p in idx)
        fout.write("\n")

    @staticmethod
    def array_writer(fout: TextIO, arr: Arr1[np.floating], level: int) -> None:
        fout.write(" " * level)
        fout.writelines(f"{p:< .16f} " for p in arr)
        fout.write("\n")


class XMLElement:
    tag: str
    data: Arr2[np.integer] | Arr2[np.floating] | Arr1[np.integer] | None
    datawriter: WRITERSIGS | None
    attribs: str
    subelems: list[XMLElement]

    def __init__(self, tag: str, **attribs: str) -> None:
        self.tag = tag
        string = ""
        for k, v in attribs.items():
            string = string + f' {k}="{v}"'
        self.attribs = string
        self.data = None
        self.datawriter = XMLWriters.int_writer
        self.subelems = []

    def create_elem(self, elem: XMLElement[T]) -> XMLElement[T]:
        self.subelems.append(elem)
        return elem

    def add_data(
        self,
        arr: Arr1[T] | Arr2[T],
        writer: WRITERSIGS[T],
    ) -> None:
        self.data = arr
        self.datawriter = writer

    def write(self, fout: TextIO, level: int = 0) -> None:
        fout.write(f"{' ' * level}<{self.tag}{self.attribs}>\n")
        for elem in self.subelems:
            elem.write(fout, level + 2)
        if self.data is not None:
            for d in self.data:
                self.datawriter(fout, d, level + 2)
        fout.write(f"{' ' * level}</{self.tag}>\n")
