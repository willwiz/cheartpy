from __future__ import annotations
import typing as tp
import numpy as np
import numpy.typing as npt


class XMLWriters:
  @staticmethod
  def PointWriter(fout:tp.TextIO, point:npt.NDArray[np.float64], level:int=0) -> None:
    fout.write(f'{" "*level}{point[0]: .16f} {point[1]: .16f} {point[2]: .16f}\n')

  @staticmethod
  def IntegerWriter(fout:tp.TextIO, id:int, level:int=0) -> None:
    fout.write(f'{" "*level}{id:d}\n')

  @staticmethod
  def FloatArrWriter(fout:tp.TextIO, arr:npt.NDArray[np.float64], level:int=0) -> None:
    fout.write(" "*(level - 1))
    for p in arr:
      fout.write(f' {p:< .16f}')
    fout.write('\n')


class XMLElement:
  data : tp.Optional[npt.NDArray] = None
  datawriter : tp.Callable[[tp.TextIO, tp.Union[int, float, npt.NDArray], tp.Optional[int]], str] = XMLWriters.PointWriter
  def __init__(self, tag:str, **attribs:str) -> None:
    self.tag = tag
    string = ""
    for k, v in attribs.items():
      string = string + f' {k}="{v}"'
    self.attribs = string
    self.subelems = list()
  def add_elem(self, elem:XMLElement):
    self.subelems.append(elem)
    return elem
  def add_data(self, arr:npt.NDArray,
               writer:tp.Callable[[tp.TextIO, tp.Union[int, float, npt.NDArray], tp.Optional[int]], str] = XMLWriters.PointWriter) -> None:
    self.data = arr
    self.datawriter = writer
  def write(self, fout:tp.TextIO, level:int=0) -> None:
    fout.write(f'{" "*level}<{self.tag}{self.attribs}>\n')
    for elem in self.subelems:
      elem.write(fout, level + 2)
    if self.data is not None:
      for d in self.data:
        self.datawriter(fout, d, level + 2)
    fout.write(f'{" "*level}</{self.tag}>\n')

