import abc
from collections.abc import Sequence
from typing import Literal, TextIO, TypedDict, Unpack

import numpy as np
from pytools.arrays import A1, A2, Arr, SAny
from pytools.result import Err, Ok

_NUM_FMT = Literal[".16f", "d", "20.16f", "24.16e"]

_gtol = 1e-8
_gmax = 10.0


def _optimal_num_fmt[S: SAny, F: np.number](arr: Arr[S, F]) -> _NUM_FMT:
    maximum = np.max(arr)
    if (maximum < _gmax) and (maximum - 0.1 >= -_gtol):
        return "20.16f"
    return "24.16e"


def create_xml_data[S: SAny, T: np.number](
    data: Arr[S, T],
    **kwargs: Unpack[_XMLKwargs],
) -> Ok[XMLData[S, T]] | Err:
    if fmt := kwargs.get("fmt"):
        _fmt = fmt
    elif np.issubdtype(data.dtype, np.floating):
        _fmt = _optimal_num_fmt(data)
    elif np.issubdtype(data.dtype, np.integer):
        _fmt = "d"
    else:
        msg = f"Only floating and integer arrays are supported, got {data.dtype}."
        return Err(TypeError(msg))
    order = kwargs.get("order")
    match data.shape, order:
        case (int(col),), None:
            _order = [0]
        case (int(), int(col)), None:
            _order = list(range(col))
        case (int(), int(col)), Sequence() | np.ndarray():
            if len(order) != col:
                msg = "order must match the number of columns in the data array. "
                return Err(ValueError(msg))
            _order = order
        case _:
            msg = f"Data must be 1D or 2D array, got {data.shape} with order {order}."
            return Err(ValueError(msg))
    return Ok(XMLData(data, order=_order, fmt=_fmt))


class XMLDataTrait(abc.ABC):
    @property
    @abc.abstractmethod
    def data(self) -> Arr[SAny, np.number]: ...
    @property
    @abc.abstractmethod
    def fmt(self) -> _NUM_FMT: ...
    @property
    @abc.abstractmethod
    def order(self) -> A1[np.integer]: ...
    @abc.abstractmethod
    def write(self, fout: TextIO, level: int = 0) -> None: ...


class _XMLKwargs(TypedDict, total=False):
    fmt: _NUM_FMT
    order: Sequence[int] | A1[np.integer]


class XMLData[S: SAny, T: np.number](XMLDataTrait):
    __slots__ = ("_data", "_fmt", "_perm")
    _data: Arr[S, T]
    _fmt: _NUM_FMT
    _perm: A1[np.integer]

    def __init__(
        self, data: Arr[S, T], order: Sequence[int] | A1[np.integer], fmt: _NUM_FMT
    ) -> None:
        self._data = data
        self._fmt = fmt
        self._perm = np.asarray(order, dtype=np.intp)

    @property
    def data(self) -> Arr[SAny, np.number]:
        return self._data

    @property
    def fmt(self) -> _NUM_FMT:
        return self._fmt

    @property
    def order(self) -> A1[np.integer]:
        return self._perm

    def write(self, fout: TextIO, level: int = 0) -> None:
        if self._data.ndim == 1:
            fout.writelines(f"{' ' * (level + 2)}{d:<{self._fmt}}\n" for d in self._data)
            return
        for arr in self._data:
            fout.write(" " * (level + 2))
            fout.write("  ".join([f"{p:<{self._fmt}}" for p in arr[self._perm]]))
            fout.write("\n")


class XMLElement:
    __slots__ = ("attribs", "data", "datawriter", "subelems", "tag")
    tag: str
    data: XMLDataTrait | None
    attribs: str
    subelems: list[XMLElement]

    def __init__(self, tag: str, **attribs: str) -> None:
        self.tag = tag
        self.attribs = " ".join([f'{k}="{v}"' for k, v in attribs.items()])
        self.data = None
        self.subelems = []

    def create_elem(self, elem: XMLElement) -> XMLElement:
        self.subelems.append(elem)
        return elem

    def add_data[S: np.number](
        self,
        arr: A2[S] | A1[S],
        **kwargs: Unpack[_XMLKwargs],
    ) -> Ok[None] | Err:
        match create_xml_data(arr, **kwargs):
            case Ok(data):
                self.data = data
                return Ok(None)
            case Err(e):
                return Err(e)

    def write(self, fout: TextIO, level: int = 0) -> None:
        fout.write(f"{' ' * level}<{self.tag} {self.attribs}>\n")
        for elem in self.subelems:
            elem.write(fout, level + 2)
        if self.data is not None:
            self.data.write(fout)
        fout.write(f"{' ' * level}</{self.tag}>\n")
