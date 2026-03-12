from typing import TYPE_CHECKING

from ._api import create_cheartmesh_from_abaqus_api
from .parsing import check_args, parse_cmdline_args

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(cmd_args: Sequence[str] | None = None) -> None:
    args = parse_cmdline_args(args=cmd_args)
    inp = check_args(args).unwrap()
    mesh = create_cheartmesh_from_abaqus_api(**inp).unwrap()
    mesh.save(inp["prefix"])


if __name__ == "__main__":
    main()
