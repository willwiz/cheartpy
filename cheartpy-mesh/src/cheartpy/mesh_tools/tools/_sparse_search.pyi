import dataclasses as dc
from typing import overload

import numpy as np
from pytools.arrays import A1, A2, DType
from pytools.result import Result
from scipy.sparse import csc_array, csr_array

@dc.dataclass(slots=True, frozen=True, init=False)
class ElementSparseMap:
    v: csc_array[np.int8]
    n_shape: int
    @overload
    def __init__(self, v: ElementSparseMap) -> None: ...
    @overload
    def __init__(self, v: csc_array[np.int8], n_shape: int) -> None: ...
    @overload
    def __init__(self, v: A2[np.integer]) -> None: ...
    def search_elements(self, elems: A2[np.integer]) -> ElementSearchResult: ...

@dc.dataclass(slots=True, frozen=True, init=False)
class ElementSearchResult:
    v: csr_array[np.int8]

    def __init__(self, result: csr_array[np.int8]) -> None: ...
    def classify_element_relation[I: np.integer = np.intc](
        self,
        *,
        mask: A1[np.bool_] | None = None,
        dtype: DType[I] = np.intc,  # noqa: PYI011
    ) -> A1[I]: ...
    def get_master_elements[I: np.integer = np.intc](
        self,
        mask: A1[np.bool_] | None = None,
        *,
        fast: bool = False,
        dtype: DType[I] = np.intc,  # noqa: PYI011
    ) -> Result[A1[I]]: ...

def convert_connectivity_to_sparse_array[I: np.integer](
    elem: A2[np.integer], nnodes: int | None = None
) -> csr_array[np.int8]: ...
def sparse_search_elements(
    sparse_map: ElementSparseMap | A2[np.integer], elems: A2[np.integer]
) -> ElementSearchResult: ...
