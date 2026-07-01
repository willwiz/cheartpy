import dataclasses as dc
from typing import TYPE_CHECKING, TextIO

import numpy as np
from pytools.result import Err, Ok, Result
from scipy.sparse import csr_matrix

if TYPE_CHECKING:
    from pathlib import Path

    from pytools.arrays import A1, DType

_SIZE_OF_HEADER = 2
_RES_LAYOUT = {"names": ("start", "end", "res", "res2"), "formats": ("i4", "i4", "f8", "f8")}
_MATRIX_LAYOUT = {"names": ("row", "col", "value"), "formats": ("i4", "i4", "f8")}


@dc.dataclass(slots=True)
class CheartMatrix[F: np.floating = np.float64, I: np.integer = np.intp]:
    n_rows: int
    n_entries: int
    row_ptr: A1[I]
    res: A1[F]
    res2: A1[F]
    matrix: csr_matrix


def _import_matrix_core[F: np.floating, I: np.integer](
    f: TextIO, *, ftype: DType[F], dtype: DType[I]
) -> Result[CheartMatrix[F]]:
    first_line = f.readline().strip().split()
    if len(first_line) != _SIZE_OF_HEADER:
        return Err(ValueError("First line must contain exactly two integers."))
    n_rows, n_entries = map(int, first_line)
    res_data = np.loadtxt(f, max_rows=n_rows, dtype=_RES_LAYOUT)
    matrix_data = np.loadtxt(f, dtype=_MATRIX_LAYOUT)
    matrix = csr_matrix(
        (matrix_data["value"], (matrix_data["row"] - 1, matrix_data["col"] - 1)),
        shape=(n_rows, n_rows),
        dtype=ftype,
    )
    if res_data["end"][-1] != n_entries + 1:
        return Err(ValueError("Number of entries does not match the last 'end' value."))
    row_ptr = np.append(res_data["start"] - 1, n_entries)
    return Ok(
        CheartMatrix(
            n_rows=n_rows,
            n_entries=n_entries,
            row_ptr=row_ptr.astype(dtype),
            res=res_data["res"].astype(ftype),
            res2=res_data["res2"].astype(ftype),
            matrix=matrix,
        )
    )


def import_cheart_matrix[F: np.floating = np.float64, I: np.integer = np.intp](
    file: Path, *, ftype: DType[F] = np.float64, dtype: DType[I] = np.intp
) -> Result[CheartMatrix[F]]:
    if not file.is_file():
        return Err(ValueError(f"File {file} does not exist."))
    with file.open("r") as f:
        return _import_matrix_core(f, ftype=ftype, dtype=dtype).next()
