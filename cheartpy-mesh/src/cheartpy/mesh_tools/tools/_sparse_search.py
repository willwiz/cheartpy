import dataclasses as dc
from typing import TYPE_CHECKING, overload

import numpy as np
from pytools.result import Err, Ok, Result
from scipy.sparse import csc_array, csr_array

if TYPE_CHECKING:
    from pytools.arrays import A1, A2, DType


@dc.dataclass(slots=True, frozen=True, init=False)
class ElementSparseMap:
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
    v: csc_array[np.int8]
    #: number of nodes per element
    n_shape: int

    @overload
    def __init__(self, v: ElementSparseMap) -> None: ...
    @overload
    def __init__(self, v: csc_array[np.int8], n_shape: int) -> None: ...
    @overload
    def __init__(self, v: A2[np.integer]) -> None: ...
    def __init__(
        self, v: ElementSparseMap | csc_array[np.int8] | A2[np.integer], n_shape: int | None = None
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

    def search_elements(self, elems: A2[np.integer]) -> ElementSearchResult:
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
        csr = convert_connectivity_to_sparse_array(elems, nnodes=self.v.shape[0])
        intersections = csr @ self.v
        res = csr_array(
            (
                intersections.data == elems.shape[-1],
                intersections.indices,
                intersections.indptr,
            ),
            shape=intersections.shape,
            dtype=np.int8,
        )
        res.eliminate_zeros()
        return ElementSearchResult(res)


@dc.dataclass(slots=True, frozen=True, init=False)
class ElementSearchResult:
    """Holds the result of searching for elements in a sparse map.

    The non-zero entries indicate that element of subtopology (row) is contained in element of
    main topology (column).

    """

    v: csr_array[np.int8]

    def __init__(self, v: csr_array[np.int8]) -> None:
        """Initialize the ElementSearchResult.

        Parameters
        ----------
        v : csr_array[I]
            A sparse matrix of search results.


        """
        object.__setattr__(self, "v", v)

    def classify_element_relation[I: np.integer = np.intc](
        self, *, mask: A1[np.bool_] | None = None, dtype: DType[I] = np.intc
    ) -> A1[I]:
        """Return an array of integers.

        Value indicate the number of time the element of the subtopology can be found in the main
        topology:

            - 0: element of subtopology is not found in main topology
            - 1: element of subtopology is likely a boundary element of main topology
            - 2: element of subtopology is likely an interior element of main topology
            - >2: element of subtopology is likely a duplicate element of main topology

        Parameters
        ----------
        mask : A1[np.bool_] | None
            A boolean array of shape (M,) where M is the number of elements in the subtopology.
            Each entry indicates whether the corresponding element is in the masked region.
            If None, all elements are considered. Defaults to None.
        dtype : DType[I]
            The data type of the output array. Defaults to np.intc.

        Returns
        -------
        A1[I]
            An array of shape (M,) where M is the number of elements in the subtopology.

        """
        if mask is None:
            return np.asarray(np.diff(self.v.indptr), dtype=dtype)
        masked_csr = _filter_csr_array_columns_by_mask(self.v, mask)
        return np.asarray(np.diff(masked_csr.indptr), dtype=dtype)

    def get_master_elements[I: np.integer = np.intc](
        self, mask: A1[np.bool_] | None = None, *, fast: bool = False, dtype: DType[I] = np.intc
    ) -> Result[A1[I]]:
        """Return the indices of the master elements in the main topology.

        Returns
        -------
        A1[I]
            An array of shape (M,) where M is the number of elements in the subtopology.
            Each entry is the index of the master element in the main topology.

        """
        if mask is None:
            return _find_master_element_from_search_result(self, fast=fast, dtype=dtype)
        return _find_master_element_in_masked_region(self, mask, fast=fast, dtype=dtype)


def convert_connectivity_to_sparse_array[I: np.integer](
    elem: A2[np.integer], nnodes: int | None = None
) -> csr_array[np.int8]:
    """Create a sparse matrix representation of the element connectivity.

    Parameters
    ----------
    elem : np.ndarray
        An array of shape (ne, n_nodes_per_elem) representing the element connectivity.
        Each row corresponds to an element and contains the global node indices for that element.

    nnodes : int | None
        The total number of nodes in the mesh. If not provided, it will be inferred from the maximum
        node index in `elem`.

    dtype : DType[I]
        The data type of the sparse matrix. Defaults to np.int8.

    Returns
    -------
    ElementSparseMap[F]
        A dataclass containing the sparse matrix representation of the element connectivity.

    """
    n_elem, n_basis = elem.shape
    n_node = nnodes or elem.max() + 1
    row_ptr = np.arange(n_elem + 1) * n_basis
    col_idx = elem.ravel()
    data = np.ones_like(col_idx, dtype=np.int8)
    return csr_array((data, col_idx, row_ptr), shape=(n_elem, n_node), dtype=np.int8)


def sparse_search_elements(
    sparse_map: ElementSparseMap | A2[np.integer], elems: A2[np.integer]
) -> ElementSearchResult:
    sparse_map = (
        sparse_map if isinstance(sparse_map, ElementSparseMap) else ElementSparseMap(sparse_map)
    )
    return sparse_map.search_elements(elems)


def _find_master_element_from_search_result[I: np.integer](
    result: ElementSearchResult, *, fast: bool = False, dtype: DType[I] = np.intc
) -> Result[A1[I]]:
    """Return the indices of the master elements in the main topology.

    Parameters
    ----------
    result : ElementSearchResult[I]
        The result of searching for elements in a sparse map.
    fast: bool
        Skip checks.
    dtype : DType[I]
        The data type of the output array. Defaults to np.intc.

    Returns
    -------
    A1[I]
        An array of shape (M,) where M is the number of elements in the subtopology.
        Each entry is the index of the master element in the main topology.

    """
    if not (fast and np.all(result.classify_element_relation() == 1)):
        msg = "Some of the elements are ambiguous. Please check with `classify_element_relation`."
        return Err(ValueError(msg))
    return Ok(np.asarray(result.v.indices[result.v.indptr[:-1]], dtype=dtype))


def _filter_csr_array_columns_by_mask[I: np.integer](
    csr: csr_array[np.int8], mask: A1[np.bool_]
) -> csr_array[np.int8]:
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
    result: ElementSearchResult,
    mask: A1[np.bool_],
    *,
    fast: bool = False,
    dtype: DType[I] = np.intc,
) -> Result[A1[I]]:
    """Return the indices of the master elements in the main topology for a masked region.

    Parameters
    ----------
    result : ElementSearchResult[I]
        The result of searching for elements in a sparse map.
    mask : A1[np.bool_]
        A boolean array of shape (M,) where M is the number of elements in the subtopology.
        Each entry indicates whether the corresponding element is in the masked region.
    fast : bool, optional
        If True, assumes that all elements in the masked region are unambiguous and returns the
        master elements directly. If False, checks for ambiguity and returns an error if any
    dtype : DType[I], optional
        The data type of the output array. Defaults to np.intc.

    Returns
    -------
    A1[I]
        An array of shape (M,) where M is the number of elements in the subtopology.
        Each entry is the index of the master element in the main topology.

    """
    masked_csr = _filter_csr_array_columns_by_mask(result.v, mask)
    masked_result = ElementSearchResult(masked_csr)
    return _find_master_element_from_search_result(masked_result, fast=fast, dtype=dtype).next()
