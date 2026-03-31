import argparse
from typing import TYPE_CHECKING

from .cylinder_core import cylinder_parser, get_cylinder_args, make_cylinder_api
from .hex_core import block_parser, get_block_args, make_block_api
from .interpolation import interp_parser, make_interp_api, parser_interp_args
from .quad_core import get_square_args, make_square_api, square_parser

if TYPE_CHECKING:
    from collections.abc import Sequence

parser = argparse.ArgumentParser("mesh")
subpar = parser.add_subparsers(dest="mode")
cube = subpar.add_parser("cube", help="3D cube mesh linear", add_help=False, parents=[block_parser])
square = subpar.add_parser(
    "square", help="2D rectangle mesh linear", add_help=False, parents=[square_parser]
)
cylinder = subpar.add_parser(
    "cylinder", help="3D cylinder mesh", add_help=False, parents=[cylinder_parser]
)
interp = subpar.add_parser(
    "interp",
    help="Interpolate variables from linear to quad mesh",
    add_help=False,
    parents=[interp_parser],
)


def main_cli(args: Sequence[str] | None = None) -> None:
    parsed_args = parser.parse_args()
    match parsed_args.mode:
        case "square":
            _args, kwargs = get_square_args(args)
            make_square_api(_args, **kwargs)
        case "cube":
            _args, kwargs = get_block_args(args)
            make_block_api(_args, **kwargs)
        case "cylinder":
            _args, kwargs = get_cylinder_args(args)
            make_cylinder_api(_args, **kwargs)
        case "interp":
            _args, kwargs = parser_interp_args(vars(parsed_args)).unwrap()
            make_interp_api(_args, **kwargs)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main_cli()
