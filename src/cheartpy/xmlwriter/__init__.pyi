__all__ = ["IVtkElementInterface", "get_element_type", "XMLWriters", "XMLElement"]
import abc
from typing import Any, TextIO, Protocol, ClassVar
from ..var_types import *
from .xmlclasses import WRITERSIGS

class IVtkElementInterface(Protocol):
    vtkelementid: ClassVar[int]
    vtksurfaceid: ClassVar[int | None]
    @staticmethod
    @abc.abstractmethod
    def write(fout: TextIO, elem: Vec[int_t], level: int = 0) -> None: ...

def get_element_type(
    nnodes: int, boundary: str | None
) -> tuple[type[IVtkElementInterface], type[IVtkElementInterface]]: ...

class XMLWriters:
    @staticmethod
    def PointWriter(fout: TextIO, point: Vec[f64], level: int = 0) -> None: ...
    @staticmethod
    def IntegerWriter(fout: TextIO, id: int, level: int = 0) -> None: ...
    @staticmethod
    def FloatArrWriter(fout: TextIO, arr: Vec[f64], level: int = 0) -> None: ...

class XMLElement:
    def __init__(self, tag: str, **attribs: str) -> None: ...
    def add_elem(self, elem: XMLElement) -> XMLElement: ...
    def add_data(
        self,
        arr: Mat[Any],
        writer: WRITERSIGS = XMLWriters.PointWriter,
    ) -> None: ...
    def write(self, fout: TextIO, level: int = 0) -> None: ...
