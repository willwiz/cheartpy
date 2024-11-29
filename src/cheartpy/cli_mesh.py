import argparse
from .mesh.make_block import parser as cube_parser, main as cube_main
from .mesh.make_square import parser as square_parser, main as square_main
from .mesh.make_cylinder import parser as cylinder_parser, main as cylinder_main

parser = argparse.ArgumentParser("mesh")
subpar = parser.add_subparsers()
cube = subpar.add_parser(
    "cube", help="3D cube mesh linear", add_help=False, parents=[cube_parser]
)
cube.set_defaults(run=cube_main)
square = subpar.add_parser(
    "cube", help="3D cube mesh linear", add_help=False, parents=[square_parser]
)
square.set_defaults(run=square_main)
cylinder = subpar.add_parser(
    "cylinder", help="3D cylinder mesh", add_help=False, parents=[cylinder_parser]
)
cylinder.set_defaults(run=cylinder_main)


def main_cli():
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main_cli()
