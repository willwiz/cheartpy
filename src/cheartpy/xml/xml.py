from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TextIO

import numpy as np

if TYPE_CHECKING:
    from arraystubs import Arr, Arr1, Arr2


class XMLData:
    __slots__ = ("data", "fmt", "order")
    data: Arr[tuple[int, ...], np.generic]
    fmt: Literal[".16f", "d"]
    order: tuple[int, ...]

    def __init__(
        self,
        data: Arr[tuple[int, ...], np.generic],
        order: tuple[int, ...] | None,
    ) -> None:
        self.data = data
        match data.dtype:
            case np.floating:
                self.fmt = ".16f"
            case np.integer:
                self.fmt = "d"
            case _:
                msg = f"Only floating and integer arrays are supported, got {data.dtype}."
                raise TypeError(msg)
        match data.shape, order:
            case (int(col),), None:
                self.order = (0,)
            case (int(), int(col)), None:
                self.order = tuple(range(col))
            case (int(), int(col)), tuple():
                if len(order) != col:
                    msg = "order must match the number of columns in the data array. "
                    raise ValueError(msg)
                self.order = order
            case _:
                msg = f"Data must be 1D or 2D array, got {data.shape} with order {order}."
                raise ValueError(msg)

    def write(self, fout: TextIO, level: int = 0) -> None:
        if self.data.ndim == 1:
            fout.writelines(f"{' ' * (level + 2)}{d:<{self.fmt}}\n" for d in self.data)
        for arr in self.data:
            fout.write(" " * (level + 2))
            fout.writelines(f"{p:<{self.fmt}} " for p in arr)
            fout.write("\n")


class XMLElement:
    __slots__ = ("attribs", "data", "datawriter", "subelems", "tag")
    tag: str
    data: XMLData | None
    attribs: str
    subelems: list[XMLElement]

    def __init__(self, tag: str, **attribs: str) -> None:
        self.tag = tag
        string = ""
        for k, v in attribs.items():
            string = string + f' {k}="{v}"'
        self.attribs = string
        self.data = None
        self.subelems = []

    def create_elem(self, elem: XMLElement) -> XMLElement:
        self.subelems.append(elem)
        return elem

    def add_data(
        self,
        arr: Arr2[np.generic] | Arr1[np.generic],
        order: tuple[int, ...] | None = None,
    ) -> None:
        self.data = XMLData(arr, order)

    def write(self, fout: TextIO, level: int = 0) -> None:
        fout.write(f"{' ' * level}<{self.tag}{self.attribs}>\n")
        for elem in self.subelems:
            elem.write(fout, level + 2)
        if self.data is None:
            return
        if self.datawriter is None:
            msg = (
                f"Data writer not set for element {self.tag}. "
                "Please set a data writer before writing."
            )
            raise ValueError(msg)
        self.data.write(fout)
        fout.write(f"{' ' * level}</{self.tag}>\n")
