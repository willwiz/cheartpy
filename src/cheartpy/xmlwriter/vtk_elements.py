from typing import TextIO
from numpy import int32
from numpy.typing import NDArray


class VtkLinearLine:
  vtkelementid = 3
  vtksurfaceid = None
  @staticmethod
  def write(fout:TextIO, elem:NDArray[int32], level:int=0) -> None:
    fout.write(" "*(level - 1))
    for j in range(2):
      fout.write(" %i" % (elem[j]-1))
    fout.write("\n")

class VtkQuadraticLine:
  vtkelementid = 21
  vtksurfaceid = None
  @staticmethod
  def write(fout:TextIO, elem:NDArray[int32], level:int=0) -> None:
    fout.write(" "*(level - 1))
    for j in range(3):
      fout.write(" %i" % (elem[j]-1))
    fout.write("\n")

class VtkBilinearTriangle:
  vtkelementid = 5
  vtksurfaceid = 3
  @staticmethod
  def write(fout:TextIO, elem:NDArray[int32], level:int=0) -> None:
    fout.write(" "*(level - 1))
    for j in range(3):
      fout.write(" %i" % (elem[j]-1))
    fout.write("\n")

class VtkBiquadraticTriangle:
  vtkelementid = 22
  vtksurfaceid = 21
  @staticmethod
  def write(fout:TextIO, elem:NDArray[int32], level:int=0) -> None:
    fout.write(" "*(level - 1))
    fout.write(" %i" % (elem[0]-1))
    fout.write(" %i" % (elem[1]-1))
    fout.write(" %i" % (elem[2]-1))
    fout.write(" %i" % (elem[3]-1))
    fout.write(" %i" % (elem[5]-1))
    fout.write(" %i" % (elem[4]-1))
    fout.write("\n")

class VtkBilinearQuadrilateral:
  vtkelementid = 9
  vtksurfaceid = 3
  @staticmethod
  def write(fout:TextIO, elem:NDArray[int32], level:int=0) -> None:
    fout.write(" "*(level - 1))
    fout.write(" %i" % (elem[0]-1))
    fout.write(" %i" % (elem[1]-1))
    fout.write(" %i" % (elem[3]-1))
    fout.write(" %i" % (elem[2]-1))
    fout.write("\n")

class VtkTrilinearTetrahedron:
  vtkelementid = 10
  vtksurfaceid = 5
  @staticmethod
  def write(fout:TextIO, elem:NDArray[int32], level:int=0) -> None:
    fout.write(" "*(level - 1))
    for j in range(4):
      fout.write(" %i" % (elem[j]-1))
    fout.write("\n")

class VtkBiquadraticQuadrilateral:
  vtkelementid = 28
  vtksurfaceid = 21
  @staticmethod
  def write(fout:TextIO, elem:NDArray[int32], level:int=0) -> None:
    fout.write(" "*(level - 1))
    fout.write(" %i" % (elem[0]-1))
    fout.write(" %i" % (elem[1]-1))
    fout.write(" %i" % (elem[3]-1))
    fout.write(" %i" % (elem[2]-1))
    fout.write(" %i" % (elem[4]-1))
    fout.write(" %i" % (elem[7]-1))
    fout.write(" %i" % (elem[8]-1))
    fout.write(" %i" % (elem[5]-1))
    fout.write(" %i" % (elem[6]-1))
    fout.write("\n")

class VtkTriquadraticTetrahedron:
  vtkelementid = 24
  vtksurfaceid = 22
  @staticmethod
  def write(fout:TextIO, elem:NDArray[int32], level:int=0) -> None:
    fout.write(" "*(level - 1))
    for j in range(10):
      if j == 6:
        fout.write(" %i" % (elem[5]-1))
      elif j == 5:
        fout.write(" %i" % (elem[6]-1))
      else:
        fout.write(" %i" % (elem[j]-1))
    fout.write("\n")

class VtkTrilinearHexahedron:
  vtkelementid = 12
  vtksurfaceid = 9
  @staticmethod
  def write(fout:TextIO, elem:NDArray[int32], level:int=0) -> None:
    fout.write(" "*(level - 1))
    fout.write(" %i" % (elem[0]-1))
    fout.write(" %i" % (elem[1]-1))
    fout.write(" %i" % (elem[5]-1))
    fout.write(" %i" % (elem[4]-1))
    fout.write(" %i" % (elem[2]-1))
    fout.write(" %i" % (elem[3]-1))
    fout.write(" %i" % (elem[7]-1))
    fout.write(" %i" % (elem[6]-1))
    fout.write("\n")

class VtkTriquadraticHexahedron:
  vtkelementid = 29
  vtksurfaceid = 28
  @staticmethod
  def write(fout:TextIO, elem:NDArray[int32], level:int=0) -> None:
    fout.write(" "*(level - 1))
    fout.write(" %i" % (elem[0]-1))
    fout.write(" %i" % (elem[1]-1))
    fout.write(" %i" % (elem[5]-1))
    fout.write(" %i" % (elem[4]-1))
    fout.write(" %i" % (elem[2]-1))
    fout.write(" %i" % (elem[3]-1))
    fout.write(" %i" % (elem[7]-1))
    fout.write(" %i" % (elem[6]-1))
    fout.write(" %i" % (elem[8]-1))
    fout.write(" %i" % (elem[15]-1))
    fout.write(" %i" % (elem[22]-1))
    fout.write(" %i" % (elem[13]-1))
    fout.write(" %i" % (elem[12]-1))
    fout.write(" %i" % (elem[21]-1))
    fout.write(" %i" % (elem[26]-1))
    fout.write(" %i" % (elem[19]-1))
    fout.write(" %i" % (elem[9]-1))
    fout.write(" %i" % (elem[11]-1))
    fout.write(" %i" % (elem[25]-1))
    fout.write(" %i" % (elem[23]-1))
    fout.write(" %i" % (elem[16]-1))
    fout.write(" %i" % (elem[18]-1))
    fout.write(" %i" % (elem[10]-1))
    fout.write(" %i" % (elem[24]-1))
    fout.write(" %i" % (elem[14]-1))
    fout.write(" %i" % (elem[20]-1))
    fout.write(" %i" % (elem[17]-1))
    fout.write("\n")
