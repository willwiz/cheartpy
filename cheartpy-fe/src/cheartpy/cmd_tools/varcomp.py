import argparse

from cheartpy.io.api import chread_d
from pytools.logging import BLogger

from ._traits import HEADER, HEADER_LEN, VarErrors, VarStats
from ._utils import compute_stats, get_variable_getter, get_variables

__all__ = ["main_cli"]


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
    "errcheck",
    type=str,
    nargs="*",
    help="Catch arguments when shell expands wildcard on variables",
)


def table_header() -> str:
    header = f"{'iter':^8}\u2016{'avg':^9}\u2016" + "|".join([f"{s:^9}" for s in HEADER])
    return header + "\n" + ("─" * len(header))


def table_row(it: int | str, res: VarStats) -> str:
    return f"{it:>8}\u2016{res}"


def main_cli() -> None:
    args = parser.parse_args()
    VarErrors.tol = args.tol
    log = BLogger("INFO")
    if len(args.errcheck) > 0:
        log.fatal("Error: Shell expanded wildcard before python")
        log.fatal("Error: Please place quotes around the offender variable")
        return
    log.disp(f"{f'{args.var1} vs {args.var2}':^{HEADER_LEN}}")
    v1, v2 = get_variables(args.var1, args.var2, root=args.folder).unwrap()
    getter = get_variable_getter(v1, v2, root=args.folder)
    log.disp(table_header())
    for k, i, j in getter:
        x = 0.0 if i is None else chread_d(i)
        y = 0.0 if j is None else chread_d(j)
        res = compute_stats(x, y)
        log.disp(table_row(k, res))


if __name__ == "__main__":
    main_cli()
