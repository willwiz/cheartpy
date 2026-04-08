from typing import TYPE_CHECKING, Unpack

from ._core import create_square_mesh

if TYPE_CHECKING:
    from ._parsing import SquareArgs, SquareKwargs


def make_square_cli(args: SquareArgs, **kwargs: Unpack[SquareKwargs]) -> None:
    mesh = create_square_mesh((args["xn"], args["yn"]), **kwargs)
    mesh.save(args["prefix"])
