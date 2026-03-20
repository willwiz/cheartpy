import dataclasses as dc
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np
    from cheartpy.elem_interfaces import VtkEnum
    from pytools.arrays import A1, A2


class _ShapeFunc(Protocol):
    def __call__[F: np.floating](self, pos: A1[F]) -> A1[F]: ...


class _ShapeFuncDeriv(Protocol):
    def __call__[F: np.floating](self, pos: A1[F]) -> A2[F]: ...


@dc.dataclass(slots=True, frozen=True)
class VtkElem:
    body: VtkEnum
    surf: VtkEnum | None
    connectivity: tuple[int, ...]
    nodes: A2[np.intc]
    ref: A2[np.float64]
    shape_func: _ShapeFunc
    shape_dfunc: _ShapeFuncDeriv
