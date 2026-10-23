import dataclasses as dc
from typing import overload

import numpy as np
from pytools.arrays import A1, A2
from pytools.result import Result
from scipy.sparse import csc_array, csr_array

@dc.dataclass(slots=True, frozen=True, init=False)
class ElementSparseMap[I: np.integer]:
    v: csc_array[I]
    n_shape: int
    @overload
    def __init__(self, v: ElementSparseMap[I]) -> None: ...
    @overload
    def __init__(self, v: csc_array[I], n_shape: int) -> None: ...
    @overload
    def __init__(self, v: A2[I]) -> None: ...
    @overload
    def search_elements(self, elems: csr_array[I]) -> ElementSearchResult[I]: ...
    @overload
    def search_elements(self, elems: A2[I]) -> ElementSearchResult[I]: ...

@dc.dataclass(slots=True, frozen=True, init=False)
class ElementSearchResult[I: np.integer]:
    v: csr_array[I]

    def __init__(self, result: csr_array[I]) -> None: ...
    def classify_element_relation(self, mask: A1[np.bool_] | None = None) -> A1[I]: ...
    def get_master_elements(self, mask: A1[np.bool_] | None = None) -> Result[A1[I]]: ...

def convert_connectivity_to_sparse_array[I: np.integer](elem: A2[I]) -> csr_array[I]: ...
def sparse_search_elements[I: np.integer](
    sparse_map: ElementSparseMap[I] | A2[I], elems: A2[I]
) -> ElementSearchResult[I]: ...
