from __future__ import annotations

import dataclasses as dc
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from cheartpy.vtk.api import guess_elem_type_from_dim

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.vtk.types import VtkType
    from pytools.arrays import A2, DType
    from pytools.parallel import ThreadMethods

    from ._trait import IFormattedName


@dc.dataclass(slots=True, frozen=True)
class ProgramArgs:
    prefix: Final[str]
    input_dir: Final[Path]
    output_dir: Final[Path]
    prog_bar: Final[bool]
    binary: Final[bool]
    compress: Final[bool]
    mpi: Final[ThreadMethods | None]
    xfile: Final[Path]
    tfile: Final[Path]
    bfile: Final[Path | None]
    space: Final[IFormattedName | None]
    disp: Final[IFormattedName | None]
    var: Final[Mapping[str, IFormattedName]]


@dc.dataclass(slots=True, frozen=True)
class ExportArgs:
    time: Final[str | int]
    space: Final[Path]
    disp: Final[Path | None]
    var: Final[Mapping[str, Path]]
    output_prefix: Final[str]
    binary: Final[bool]
    compress: Final[bool]


class ParaviewTopology[F: np.floating, I: np.integer]:
    __slots__ = ["_ft", "_fx", "nc", "ne", "vtkelementtype", "vtksurfacetype"]

    _ft: Final[A2[I]]
    _fx: Final[A2[F]]
    ne: Final[int]
    nc: Final[int]
    vtkelementtype: Final[VtkType]
    vtksurfacetype: Final[VtkType | None]

    def __init__(
        self, x: A2[F], tfile: Path | str, bfile: Path | None, *, dtype: DType[I] = np.intc
    ) -> None:
        ############################################################################################
        # read topology and get number of elements, number of nodes per elements
        self._ft = np.loadtxt(tfile, skiprows=1, dtype=dtype) - 1
        if self._ft.ndim == 1:
            self._ft = self._ft[:, np.newaxis]
        self._fx = x
        self.ne = self._ft.shape[0]
        self.nc = self._ft.shape[1]
        # guess the VTK element type
        match bfile:
            case Path():
                with Path(bfile).open("r") as f:
                    _ = next(f)  # skip header
                    bdim = len(next(f).strip().split()) - 2
            case None:
                bdim = None
        vtk = guess_elem_type_from_dim(self.nc, bdim).unwrap()
        self.vtkelementtype, self.vtksurfacetype = vtk.body, vtk.surf

    # def __setitem__(self, index: int, data: A1[I]) -> None:
    #     self._ft[index] = data

    # def __getitem__(self, index: int) -> A1[I]:
    #     return self._ft[index]

    @property
    def x(self) -> A2[F]:
        return self._fx

    @property
    def t(self) -> A2[I]:
        return self._ft


@dc.dataclass(slots=True)
class VariableCache[F: np.floating, I: np.integer]:
    top: Final[ParaviewTopology[F, I]]
    time: str | int
    fx: Path | None
    fd: Path | None
    fv: dict[str, Path]
    ftype: Final[DType[F]]
    dtype: Final[DType[I]]


@dc.dataclass(slots=True, frozen=True)
class XMLDataInputs[F: np.floating, I: np.integer]:
    prefix: Final[str]
    path: Final[Path]
    time: Final[str | int]
    top: Final[ParaviewTopology[F, I]]
    x: Final[Path | None]
    u: Final[Path | None]
    var: Final[Mapping[str, A2[F]]] | Final[Mapping[str, Path]]
    compress: Final[bool]
