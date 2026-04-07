from typing import TYPE_CHECKING

from ._api import varcomp_api
from ._parser import parse_cmdline_args

if TYPE_CHECKING:
    from collections.abc import Sequence


def varcomp_cli(cmd_args: Sequence[str] | None = None) -> None:
    args, kwargs = parse_cmdline_args(cmd_args).unwrap()
    varcomp_api(args["var_1"], args["var_2"], **kwargs)
