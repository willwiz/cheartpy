import argparse

from .hex_core.api import create_hex_mesh

parser = argparse.ArgumentParser("block", description="Make a cube")
parser.add_argument(
    "--prefix",
    "-p",
    type=str,
    default="cube",
    help="Prefix for saved file.",
)
parser.add_argument(
    "--size",
    "-s",
    type=float,
    nargs=3,
    default=[1, 1, 1],
    metavar=("Lx", "Ly", "Lz"),
    help="starting corner should be shifted by",
)
parser.add_argument(
    "--offset",
    "-o",
    type=float,
    nargs=3,
    default=[0, 0, 0],
    metavar=("x(0)", "y(0)", "z(0)"),
    help="starting corner should be shifted by",
)
parser.add_argument("xn", type=int, help="number of elements in x")
parser.add_argument("yn", type=int, help="number of elements in y")
parser.add_argument("zn", type=int, help="number of elements in z")


def main(args: argparse.Namespace) -> None:
    mesh = create_hex_mesh(
        (args.xn, args.yn, args.zn),
        tuple(args.size),
        tuple(args.offset),
    )
    mesh.save(args.prefix)


# ----  Here beging the main program  ---------------------------------------
# Get the command line arguments
if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
