import dataclasses as dc
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from cheartpy.vtk.api import guess_elem_type_from_dim

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.vtk.types import VtkType
    from pytools.arrays import A1, A2, DType

    from ._trait import IFormattedName


@dc.dataclass(slots=True, frozen=True)
class ProgramArgs:
    prefix: Final[str]
    input_dir: Final[Path]
    output_dir: Final[Path]
    prog_bar: Final[bool]
    binary: Final[bool]
    compress: Final[bool]
    cores: Final[int]
    tfile: Final[Path]
    bfile: Final[Path | None]
    xfile: Final[IFormattedName]
    disp: Final[IFormattedName | None]
    var: Final[Mapping[str, IFormattedName]]


class CheartTopology[I: np.integer]:
    __slots__ = ["_ft", "nc", "ne", "vtkelementtype", "vtksurfacetype"]

    _ft: A2[I]
    ne: int
    nc: int
    vtkelementtype: VtkType
    vtksurfacetype: VtkType | None

    def __init__(self, tfile: Path | str, bfile: Path | None, *, dtype: DType[I] = np.intc) -> None:
        ################################################################################################
        # read topology and get number of elements, number of nodes per elements
        self._ft = np.loadtxt(tfile, skiprows=1, dtype=dtype) - 1
        if self._ft.ndim == 1:
            self._ft = self._ft[:, np.newaxis]
        self.ne = self._ft.shape[0]
        self.nc = self._ft.shape[1]
        # guess the VTK element type
        # bilinear triangle
        match bfile:
            case Path():
                with Path(bfile).open("r") as f:
                    _ = next(f)  # skip header
                    bdim = len(next(f).strip().split()) - 2
            case None:
                bdim = None
        vtk = guess_elem_type_from_dim(self.nc, bdim).unwrap()
        self.vtkelementtype, self.vtksurfacetype = vtk.body, vtk.surf

    def __setitem__(self, index: int, data: A1[I]) -> None:
        self._ft[index] = data

    def __getitem__(self, index: int) -> int | A1[I]:
        return self._ft[index]

    def get_data(self) -> A2[I]:
        return self._ft


@dc.dataclass(slots=True)
class VariableCache[F: np.floating, I: np.integer]:
    top: Final[CheartTopology[I]]
    t: str | int
    space_i: Path
    disp_i: Path | None
    space: A2[F]
    disp: A2[F]
    x: A2[F]
    var_i: dict[str, Path] = dc.field(default_factory=dict[str, Path])
    var: dict[str, A2[F]] = dc.field(default_factory=dict[str, "A2[F]"])


@dc.dataclass(slots=True)
class InputArguments:
    space: str | A2[np.float64]
    disp: str | None
    var: dict[str, str]
    prefix: str
