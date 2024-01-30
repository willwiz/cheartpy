import argparse
from cheartpy.meshing.interp_l2q_tet import parser as tet_parser, main_cli as tet_main


parser = argparse.ArgumentParser("interpDfiles")
subpar = parser.add_subparsers()
tet = subpar.add_parser(
    "tet",
    help="interp tetrahedrons from linear to quad",
    add_help=False,
    parents=[tet_parser],
)
tet.set_defaults(run=tet_main)


def main_cli():
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main_cli()
