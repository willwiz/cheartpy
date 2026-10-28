import dataclasses as dc
from pathlib import Path

import numpy as np
from pytools.arrays import A1, DType
from pytools.result import Result
from scipy.sparse import csr_matrix

@dc.dataclass(slots=True)
class CheartMatrix[F: np.floating = np.float64, I: np.integer = np.intp]:
    n_rows: int
    n_entries: int
    row_ptr: A1[I]
    res: A1[F]
    res2: A1[F]
    matrix: csr_matrix

def import_cheart_matrix[F: np.floating = np.float64, I: np.integer = np.intp](
    file: Path, *, ftype: DType[F] = np.float64, dtype: DType[I] = np.intp
) -> Result[CheartMatrix[F, I]]: ...
