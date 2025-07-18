from collections.abc import Sequence

from .api import create_cheartmesh_from_abaqus
from .parser import check_args, parser


def main(cmd_args: Sequence[str] | None = None) -> None:
    args = parser.parse_args(args=cmd_args)
    inp = check_args(args)
    create_cheartmesh_from_abaqus(inp)


if __name__ == "__main__":
    main()
