from typing import Literal, TextIO

import numpy as np
from arraystubs import Arr, Arr1, Arr2


class XMLData[T: np.number]:
    __slots__ = ("data", "fmt", "order")
    data: Arr[tuple[int, ...], T]
    fmt: Literal[".16f", "d"]
    order: list[int]

    def __init__(
        self,
        data: Arr[tuple[int, ...], T],
        order: tuple[int, ...] | None,
    ) -> None:
        self.data = data
        if np.issubdtype(data.dtype, np.floating):
            self.fmt = ".16f"
        elif np.issubdtype(data.dtype, np.integer):
            self.fmt = "d"
        else:
            msg = f"Only floating and integer arrays are supported, got {data.dtype}."
            raise TypeError(msg)
        match data.shape, order:
            case (int(col),), None:
                self.order = [0]
            case (int(), int(col)), None:
                self.order = list(range(col))
            case (int(), int(col)), tuple():
                if len(order) != col:
                    msg = "order must match the number of columns in the data array. "
                    raise ValueError(msg)
                self.order = list(order)
            case _:
                msg = f"Data must be 1D or 2D array, got {data.shape} with order {order}."
                raise ValueError(msg)

    def write(self, fout: TextIO, level: int = 0) -> None:
        if self.data.ndim == 1:
            fout.writelines(f"{' ' * (level + 2)}{d:<{self.fmt}}\n" for d in self.data)
            return
        for arr in self.data:
            fout.write(" " * (level + 2))
            fout.writelines(f"{p:<{self.fmt}} " for p in arr[self.order])
            fout.write("\n")


class XMLElement:
    __slots__ = ("attribs", "data", "datawriter", "subelems", "tag")
    tag: str
    data: XMLData[np.number] | None
    attribs: str
    subelems: list["XMLElement"]

    def __init__(self, tag: str, **attribs: str) -> None:
        self.tag = tag
        string = ""
        for k, v in attribs.items():
            string = string + f' {k}="{v}"'
        self.attribs = string
        self.data = None
        self.subelems = []

    def create_elem(self, elem: "XMLElement") -> "XMLElement":
        self.subelems.append(elem)
        return elem

    def add_data[S: np.number](
        self,
        arr: Arr2[S] | Arr1[S],
        order: tuple[int, ...] | None = None,
    ) -> None:
        self.data = XMLData(arr, order)

    def write(self, fout: TextIO, level: int = 0) -> None:
        fout.write(f"{' ' * level}<{self.tag}{self.attribs}>\n")
        for elem in self.subelems:
            elem.write(fout, level + 2)
        if self.data is not None:
            self.data.write(fout)
        fout.write(f"{' ' * level}</{self.tag}>\n")
