from typing import TYPE_CHECKING

from .api import create_cheartmesh_from_abaqus
from .parser import check_args, parser

if TYPE_CHECKING:
    from collections.abc import Sequence


def main(cmd_args: Sequence[str] | None = None) -> None:
    args = parser.parse_args(args=cmd_args)
    inp = check_args(args)
    create_cheartmesh_from_abaqus(inp)


if __name__ == "__main__":
    main()
