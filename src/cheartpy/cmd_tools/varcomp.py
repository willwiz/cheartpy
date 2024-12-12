__all__ = ["main"]
from ..tools.basiclogging import BLogger
from ..cheart_mesh.io import CHRead_d
from .traits import HEADER_LEN, HEADER, VarErrors, VarStats
from .funcs import get_variable_getter, compute_stats
import argparse


parser = argparse.ArgumentParser(
    description="""
    Compare the values of two arrays imported from  file
    """,
)
parser.add_argument(
    "--folder",
    "-f",
    type=str,
    default=None,
    help="OPTIONAL: Relative path to data directory",
)
parser.add_argument(
    "--tol",
    type=float,
    default=1e-10,
    help="OPTIONAL: tolerance for values to be highlighted",
)
parser.add_argument("var1", type=str, help="get name or prefix of first files")
parser.add_argument("var2", type=str, help="get name or prefix of second files")
parser.add_argument(
    "_errcheck",
    type=str,
    nargs="*",
    help="Catch arguments when shell expands wildcard on variables",
)


def print_table_header() -> None:
    print(f"{r"#":^8}|", end="")
    print(f"{"Mag":^8}||", end="")
    print("|".join([f"{s:^10}" for s in HEADER]), end="\n")
    print(HEADER_LEN * "─")


def print_table_row(iter: int | str, res: VarStats) -> None:
    print(f"{iter:>8}", end="|")
    print(str(res), end="\n")


def main() -> None:
    args = parser.parse_args()
    VarErrors.tol = args.tol
    LOG = BLogger("INFO")
    if len(args._errcheck) > 0:
        LOG.fatal(f"Error: Shell expanded wildcard before python")
        LOG.fatal(f"Error: Please place quotes around the offender variable")
        return
    LOG.disp(f"{f"{args.var1} vs {args.var2}":^{HEADER_LEN}}")
    getter = get_variable_getter(args.var1, args.var2, root=args.folder)
    print_table_header()
    for k, i, j in getter:
        x = 0.0 if i is None else CHRead_d(i)
        y = 0.0 if j is None else CHRead_d(j)
        res = compute_stats(x, y)
        print_table_row(k, res)


if __name__ == "__main__":
    main()
