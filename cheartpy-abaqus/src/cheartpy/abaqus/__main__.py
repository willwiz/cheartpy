from typing import TYPE_CHECKING

from pytools.logging import get_logger

from .__logging__ import compose_header, format_input_kwargs, header_guard
from ._api import create_cheartmesh_from_abaqus_api
from .parsing import check_args, parse_cmdline_args

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(cmd_args: Sequence[str] | None = None) -> None:
    args = parse_cmdline_args(args=cmd_args)
    inp = check_args(args).unwrap()
    log = get_logger(level=inp.get("log_level"))
    log.info(*compose_header(), *format_input_kwargs(**inp))
    mesh = create_cheartmesh_from_abaqus_api(**inp).unwrap()
    mesh.save(inp["prefix"])
    log.info(header_guard(" COMPLETE! "), "")


if __name__ == "__main__":
    main()
