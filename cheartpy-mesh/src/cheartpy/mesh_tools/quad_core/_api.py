from typing import TYPE_CHECKING, Unpack

from ._core import create_square_mesh

if TYPE_CHECKING:
    import numpy as np

    from cheartpy.mesh import CheartMesh

    from ._parsing import SquareArgs, SquareKwargs


def make_square_api(
    dim: SquareArgs, **kwargs: Unpack[SquareKwargs]
) -> CheartMesh[np.float64, np.intc]:
    mesh = create_square_mesh(
        (dim["xn"], dim["yn"]), **{k: kwargs[k] for k in ["shape", "offset"] if k in kwargs}
    )
    if prefix := kwargs.get("prefix"):
        mesh.save(prefix)
    return mesh
