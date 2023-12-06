import argparse
from cheartpy.meshing.make_grid import parser as cube_parser, main as cube_main
from cheartpy.meshing.make_cylinder import (
    parser as cylinder_parser,
    main as cylinder_main,
)
from cheartpy.meshing.make_arc import parser as arc_parser, main as arc_main
from cheartpy.meshing.make_nonlinear_timesteps import (
    parser as time_parser,
    main_cli as time_main,
)

parser = argparse.ArgumentParser("mesh")
subpar = parser.add_subparsers()
cube = subpar.add_parser(
    "cube", help="3D cube mesh linear", add_help=False, parents=[cube_parser]
)
cube.set_defaults(run=cube_main)
cylinder = subpar.add_parser(
    "cylinder", help="3D cylinder mesh", add_help=False, parents=[cylinder_parser]
)
cylinder.set_defaults(run=cylinder_main)
arc = subpar.add_parser("arc", help="3D arc mesh", add_help=False, parents=[arc_parser])
arc.set_defaults(run=arc_main)
cylinder.set_defaults(run=cylinder_main)
time = subpar.add_parser(
    "time", help="nonlinear time steps", add_help=False, parents=[time_parser]
)
time.set_defaults(run=time_main)


def main_cli():
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main_cli()
