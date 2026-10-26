from pathlib import Path
from typing import TYPE_CHECKING, Unpack, overload

from pytools.logging import get_logger

from cheartpy.io import chread_d

from ._stats import compute_stats, table_row
from ._var_getters import create_argument_list, parse_variable_input

if TYPE_CHECKING:
    from ._traits import VarCompAPIKwargs

HEADER = ["mean", "std", "min", "min pos", "max", "max pos", "bias"]
HEADER_LEN = 8 + 11 + len(HEADER) * 11


def table_header() -> str:
    header = f"{'iter':^8}\u2016{'mag':^9} ::" + "|".join([f"{s:^9}" for s in HEADER])
    return header + "\n" + ("─" * len(header))


@overload
def varcomp_api(var1: str, **kwargs: Unpack[VarCompAPIKwargs]) -> None: ...
@overload
def varcomp_api(var1: str, var2: str, **kwargs: Unpack[VarCompAPIKwargs]) -> None: ...
def varcomp_api(var1: str, var2: str | None = None, **kwargs: Unpack[VarCompAPIKwargs]) -> None:
    """Compare two variables and print the results to the console.

    Parameters
    ----------
    var1 : str
        The first variable to compare.
    var2 : str, optional
        The second variable to compare. Default to var1 if not provided.
    **kwargs : Unpack[VarCompAPIKwargs]
        Additional keyword arguments for configuration.

    Kwargs
    ------
    root_1 : Path
        The root directory for the first variable. Defaults to the current working directory.
    root_2 : Path
        The root directory for the second variable. Optional. Defaults to root_1.
    log_level : LogLevel
        The logging level for the output. Defaults to "INFO". Choices:

        - "DEBUG"
        - "INFO"
        - "BRIEF"
        - "WARN"
        - "ERROR"
        - "FATAL"
        - "NULL"

    """
    log = get_logger(level=(kwargs.get("log_level") or "INFO"))
    var2 = var2 or var1
    root_1 = kwargs.get("root_1") or kwargs.get("root_2") or Path.cwd()
    root_2 = kwargs.get("root_2") or kwargs.get("root_1") or Path.cwd()
    v1 = parse_variable_input(var1, root_1).unwrap()
    v2 = parse_variable_input(var2, root_2).unwrap()
    arg_list = create_argument_list(v1, v2)
    log.info(f"{f'{v1.prefix} vs {v2.prefix}':^{HEADER_LEN}}")
    log.disp(table_header())
    for i, x, y in arg_list:
        x_data = chread_d(x)
        y_data = chread_d(y)
        res = compute_stats(x_data, y_data).unwrap()
        log.disp(table_row(i, res))
