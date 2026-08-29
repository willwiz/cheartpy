from collections.abc import Collection, Mapping, Sequence
from typing import Literal, overload

import numpy as np
from pytools.arrays import A1, A2
from pytools.result import Result

from ._types import ElemSearchMap

def build_element_searchmap[I: np.integer](elements: Mapping[int, A1[I]]) -> ElemSearchMap: ...
@overload
def find_elements[I: np.integer](top: ElemSearchMap, nodes: A1[I]) -> Result[int]: ...
@overload
def find_elements[I: np.integer](
    top: ElemSearchMap, nodes: A1[I], *, unique: Literal[True]
) -> Result[int]: ...
@overload
def find_elements[I: np.integer](
    top: ElemSearchMap, nodes: A1[I], *, unique: Literal[False]
) -> Result[Collection[int]]: ...
@overload
def find_elements[I: np.integer](
    top: A2[I], nodes: A1[I], *, keys: A1[I] | None = None
) -> Result[int]: ...
@overload
def find_elements[I: np.integer](
    top: A2[I], nodes: A1[I], *, unique: Literal[True], keys: A1[I] | None = None
) -> Result[int]: ...
@overload
def find_elements[I: np.integer](
    top: A2[I], nodes: A1[I], *, unique: Literal[False], keys: A1[I] | None = None
) -> Result[Collection[int]]: ...
@overload
def find_elements[I: np.integer](top: ElemSearchMap, nodes: A2[I]) -> Result[Sequence[int]]: ...
@overload
def find_elements[I: np.integer](
    top: ElemSearchMap, nodes: A2[I], *, unique: Literal[True]
) -> Result[Sequence[int]]: ...
@overload
def find_elements[I: np.integer](
    top: ElemSearchMap, nodes: A2[I], *, unique: Literal[False]
) -> Result[Sequence[Collection[int]]]: ...
@overload
def find_elements[I: np.integer](
    top: A2[I], nodes: A2[I], *, keys: A1[I] | None = None
) -> Result[Sequence[int]]: ...
@overload
def find_elements[I: np.integer](
    top: A2[I], nodes: A2[I], *, unique: Literal[True], keys: A1[I] | None = None
) -> Result[Sequence[int]]: ...
@overload
def find_elements[I: np.integer](
    top: A2[I], nodes: A2[I], *, unique: Literal[False], keys: A1[I] | None = None
) -> Result[Sequence[Collection[int]]]: ...
