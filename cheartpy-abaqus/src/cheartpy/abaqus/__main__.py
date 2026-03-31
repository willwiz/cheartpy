from typing import TYPE_CHECKING

from pytools.logging import get_logger

from .__logging__ import compose_header, format_input_kwargs, header_guard
from ._api import create_cheartmesh_from_abaqus_api
from .parsing import parse_cmdline_args

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(cmd_args: Sequence[str] | None = None) -> None:
    args, kwargs = parse_cmdline_args(args=cmd_args)
    log = get_logger(level=kwargs.get("log_level"))
    log.info(*compose_header(), *format_input_kwargs(args["files"], **kwargs))
    create_cheartmesh_from_abaqus_api(args["files"], **kwargs).unwrap()
    log.info(header_guard(" COMPLETE! "), "")


if __name__ == "__main__":
    main()
