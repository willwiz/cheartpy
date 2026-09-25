from __future__ import annotations

import dataclasses as dc
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np
from cheartpy.elem_interfaces import CheartEnum, get_boundary_element, guess_element_from_dim

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.search._varible_index import VariableType
    from pytools.arrays import A2, DType
    from pytools.parallel import ThreadMethods


@dc.dataclass(slots=True, frozen=True)
class ProgramArgs:
    prefix: str
    input_dir: Path
    output_dir: Path
    prog_bar: bool
    binary: bool
    compress: bool
    mpi: ThreadMethods | None
    xfile: Path
    tfile: Path
    bfile: Path | None
    space: VariableType | None
    disp: VariableType | None
    cell_var: Mapping[str, VariableType]
    point_var: Mapping[str, VariableType]


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
    __slots__ = ["_ft", "_fx", "elementtype", "nc", "ne", "surfacetype"]

    _ft: Final[A2[I]]
    _fx: Final[A2[F]]
    ne: Final[int]
    nc: Final[int]
    elementtype: Final[CheartEnum]
    surfacetype: Final[CheartEnum | None]

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
        self.elementtype = guess_element_from_dim(self.nc, bdim, "Cheart").unwrap()
        self.surfacetype = get_boundary_element(self.elementtype)

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
    time: int | tuple[int, int]
    fx: Path | None
    fd: Path | None
    fv: dict[str, Path]
    fc: dict[str, Path]
    ftype: Final[DType[F]]
    dtype: Final[DType[I]]


@dc.dataclass(slots=True, frozen=True)
class XMLDataInputs[F: np.floating, I: np.integer]:
    prefix: Final[str]
    path: Final[Path]
    time: Final[int | tuple[int, int]]
    top: Final[ParaviewTopology[F, I]]
    x: Final[Path | None]
    u: Final[Path | None]
    point_var: Final[Mapping[str, A2[F]]] | Final[Mapping[str, Path]]
    cell_var: Final[Mapping[str, A2[F]]] | Final[Mapping[str, Path]]
    compress: Final[bool]
    ftype: Final[DType[F]]
    dtype: Final[DType[I]]
