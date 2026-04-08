import argparse
from typing import TYPE_CHECKING

from .cylinder_core import cylinder_parser, make_cylinder_cli, parse_cylinder_args
from .hex_core import block_parser, make_block_cli, parse_block_args
from .interpolation import interp_parser, make_interp_cli, parser_interp_args
from .quad_core import make_square_cli, parse_square_args, square_parser

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
    parsed_args = parser.parse_args(args)
    match parsed_args.mode:
        case "square":
            _args, kwargs = parse_square_args(vars(parsed_args)).unwrap()
            make_square_cli(_args, **kwargs)
        case "cube":
            _args, kwargs = parse_block_args(vars(parsed_args)).unwrap()
            make_block_cli(_args, **kwargs)
        case "cylinder":
            _args, kwargs = parse_cylinder_args(vars(parsed_args)).unwrap()
            make_cylinder_cli(_args, **kwargs)
        case "interp":
            _args, kwargs = parser_interp_args(vars(parsed_args)).unwrap()
            make_interp_cli(_args, **kwargs)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main_cli()
