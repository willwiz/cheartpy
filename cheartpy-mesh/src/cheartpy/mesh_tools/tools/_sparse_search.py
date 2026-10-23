import dataclasses as dc
from typing import TYPE_CHECKING, overload

import numpy as np
from pytools.result import Err, Ok, Result
from scipy.sparse import csc_array, csr_array

if TYPE_CHECKING:
    from pytools.arrays import A1, A2


@dc.dataclass(slots=True, frozen=True, init=False)
class ElementSparseMap[I: np.integer]:
    """Holds a transposed sparse matrix representation of the element connectivity.

    let:
        N: number of nodes in mesh
        E: number of elements in mesh
        M: number of elements in subtopology
        A [E, N]: sparse matrix representation of the element connectivity of the main topology
        B [M, N]: sparse matrix representation of the element connectivity of the subtopology

    B [M, N] @ A.T [N, E] => [M, E] is a sparse matrix where the non-zero entries of each row
    correspond element of subtopology, column corresponds to element of main topology, and the
    non-zero entries indicate the number of nodes shared between the two elements from the two
    two topologies.

    """

    #: dimensions: [N, E]
    v: csc_array[I]
    #: number of nodes per element
    n_shape: int

    @overload
    def __init__(self, v: ElementSparseMap[I]) -> None: ...
    @overload
    def __init__(self, v: csc_array[I], n_shape: int) -> None: ...
    @overload
    def __init__(self, v: A2[I]) -> None: ...
    def __init__(
        self, v: ElementSparseMap[I] | csc_array[I] | A2[I], n_shape: int | None = None
    ) -> None:
        """Initialize the ElementSparseMap.

        Signatures
        ----------
        ElementSparseMap(self, v: ElementSparseMap[I]) -> None
        ElementSparseMap(self, v: csc_array[I], n_shape: int) -> None
        ElementSparseMap(self, v: A2[I]) -> None  Parameters

        Parameters
        ----------
        v : ElementSparseMap[I] | csc_array[I] | A2[I]
            The sparse matrix representation of the element connectivity or an array of element
            connectivity.
        n_shape : int | None
            The number of nodes per element. Required if `v` is a csc_array and not provided.
            Defaults to None.

        """
        match v:
            case ElementSparseMap():
                object.__setattr__(self, "v", v.v)
                object.__setattr__(self, "n_shape", v.n_shape)
            case csc_array() if n_shape is not None:
                object.__setattr__(self, "v", v)
                object.__setattr__(self, "n_shape", n_shape)
            case np.ndarray():
                object.__setattr__(self, "v", convert_connectivity_to_sparse_array(v).T)
                object.__setattr__(self, "n_shape", int(v.shape[-1]))
            case _:
                msg = "Invalid input combination. Please respect overloads"
                raise ValueError(msg)

    @overload
    def search_elements(self, elems: csr_array[I]) -> ElementSearchResult[I]: ...
    @overload
    def search_elements(self, elems: A2[I]) -> ElementSearchResult[I]: ...
    def search_elements(self, elems: csr_array[I] | A2[I]) -> ElementSearchResult[I]:
        """Search for elements in the sparse map that contain the given elements.

        Parameters
        ----------
        elems : csr_array[I] | np.ndarray[I, ndim=2]
            An array of shape (ne, n_nodes_per_elem) representing the element connectivity.
            Each row corresponds to an element and contains the global node indices.

        Returns
        -------
        csr_array[I]
            A sparse matrix where each row corresponds to an element in `elems` and each column
            corresponds to an element in the sparse map. The non-zero entries indicate the number
            of nodes shared between the two elements.

        """
        elems = (
            elems if isinstance(elems, csr_array) else convert_connectivity_to_sparse_array(elems)
        )
        res = elems @ self.v
        res.data = (res.data == self.n_shape).astype(res.dtype)
        res.eliminate_zeros()
        return ElementSearchResult(res)


@dc.dataclass(slots=True, frozen=True, init=False)
class ElementSearchResult[I: np.integer]:
    """Holds the result of searching for elements in a sparse map.

    The non-zero entries indicate that element of subtopology (row) is contained in element of
    main topology (column).

    """

    v: csr_array[I]

    def __init__(self, result: csr_array[I]) -> None:
        """Initialize the ElementSearchResult.

        Parameters
        ----------
        result : csr_array[I]
            A sparse matrix of search results.


        """
        object.__setattr__(self, "result", result)

    def classify_element_relation(self, mask: A1[np.bool_] | None = None) -> A1[I]:
        """Return an array of integers.

        Value indicate the number of time the element of the subtopology can be found in the main
        topology:
            0: element of subtopology is not found in main topology
            1: element of subtopology is likely a boundary element of main topology
            2: element of subtopology is likely an interior element of main topology
            >2: element of subtopology is likely a duplicate element of main topology

        Returns
        -------
        A1[I]
            An array of shape (M,) where M is the number of elements in the subtopology.

        """
        if mask is None:
            return np.asarray(np.diff(self.v.indptr), dtype=self.v.dtype)
        masked_csr = _filter_csr_array_columns_by_mask(self.v, mask)
        return np.asarray(np.diff(masked_csr.indptr), dtype=masked_csr.dtype)

    def get_master_elements(self, mask: A1[np.bool_] | None = None) -> Result[A1[I]]:
        """Return the indices of the master elements in the main topology.

        Returns
        -------
        A1[I]
            An array of shape (M,) where M is the number of elements in the subtopology.
            Each entry is the index of the master element in the main topology.

        """
        if mask is None:
            return _find_master_element_from_search_result(self)
        return _find_master_element_in_masked_region(self, mask)


def convert_connectivity_to_sparse_array[I: np.integer](elem: A2[I]) -> csr_array[I]:
    """Create a sparse matrix representation of the element connectivity.

    Parameters
    ----------
    elem : np.ndarray
        An array of shape (ne, n_nodes_per_elem) representing the element connectivity.
        Each row corresponds to an element and contains the global node indices for that element.

    Returns
    -------
    ElementSparseMap[F]
        A dataclass containing the sparse matrix representation of the element connectivity.

    """
    n_elem, n_basis = elem.shape
    n_node = elem.max() + 1
    row_ptr = np.arange(n_elem + 1, dtype=elem.dtype) * n_basis
    col_idx = elem.ravel()
    data = np.ones_like(col_idx, dtype=elem.dtype)
    return csr_array((data, col_idx, row_ptr), shape=(n_elem, n_node), dtype=elem.dtype)


def sparse_search_elements[I: np.integer](
    sparse_map: ElementSparseMap[I] | A2[I], elems: A2[I]
) -> ElementSearchResult[I]:
    sparse_map = (
        sparse_map if isinstance(sparse_map, ElementSparseMap) else ElementSparseMap(sparse_map)
    )
    return sparse_map.search_elements(elems)


def _find_master_element_from_search_result[I: np.integer](
    result: ElementSearchResult[I],
) -> Result[A1[I]]:
    """Return the indices of the master elements in the main topology.

    Parameters
    ----------
    result : ElementSearchResult[I]
        The result of searching for elements in a sparse map.

    Returns
    -------
    A1[I]
        An array of shape (M,) where M is the number of elements in the subtopology.
        Each entry is the index of the master element in the main topology.

    """
    if not np.all(result.classify_element_relation() == 1):
        msg = "Some of the elements are ambiguous. Please check with `classify_element_relation`."
        return Err(ValueError(msg))
    return Ok(np.asarray(result.v.indices[result.v.indptr[:-1]], dtype=result.v.dtype))


def _filter_csr_array_columns_by_mask[I: np.integer](
    csr: csr_array[I], mask: A1[np.bool_]
) -> csr_array[I]:
    """Filter the columns of a CSR array based on a boolean mask.

    Parameters
    ----------
    csr : csr_array[I]
        The input CSR array to be filtered.
    mask : A1[np.bool_]
        A boolean array of columns to keey.

    Returns
    -------
    csr_array[I]
        A new CSR array with only the columns that correspond to True values in the mask.

    """
    masked_data = csr.data * mask[csr.indices]
    masked_csr = csr_array((masked_data, csr.indices, csr.indptr), shape=csr.shape, dtype=csr.dtype)
    masked_csr.eliminate_zeros()
    return masked_csr


def _find_master_element_in_masked_region[I: np.integer](
    result: ElementSearchResult[I], mask: A1[np.bool_]
) -> Result[A1[I]]:
    """Return the indices of the master elements in the main topology for a masked region.

    Parameters
    ----------
    result : ElementSearchResult[I]
        The result of searching for elements in a sparse map.
    mask : A1[np.bool_]
        A boolean array of shape (M,) where M is the number of elements in the subtopology.
        Each entry indicates whether the corresponding element is in the masked region.

    Returns
    -------
    A1[I]
        An array of shape (M,) where M is the number of elements in the subtopology.
        Each entry is the index of the master element in the main topology.

    """
    masked_csr = _filter_csr_array_columns_by_mask(result.v, mask)
    masked_result = ElementSearchResult(masked_csr)
    return _find_master_element_from_search_result(masked_result).next()
