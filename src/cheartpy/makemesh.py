import argparse
from .meshing.make_grid import (
    parser as cube_parser,
    main as cube_main,
)

parser = argparse.ArgumentParser('mesh')
subpar = parser.add_subparsers()
cube = subpar.add_parser('cube', help='3D cube mesh linear', add_help=False, parents=[cube_parser])
cube.set_defaults(run=cube_main)


def main_cli():
    args = parser.parse_args()
    args.run(args)


if __name__=="__main__":
    main_cli()
