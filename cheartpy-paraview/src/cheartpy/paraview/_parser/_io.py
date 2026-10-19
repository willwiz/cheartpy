import argparse
from pathlib import Path
from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Sequence


class SetBothAction(argparse.Action):
    @override
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[str] | None,
        option_string: str | None = None,
    ) -> None:
        namespace.input_dir = values
        namespace.output_dir = values


io_parser = argparse.ArgumentParser(add_help=False)
_iogroup = io_parser.add_argument_group(title="IO")
_iogroup.add_argument(
    "--prefix",
    "-p",
    type=str,
    default=None,
    help=(
        "OPTIONAL: supply a prefix name to be used for the exported vtu files."
        ' Prefix fallback is .prefix > .output_dir > .input_dir > "paraview".'
    ),
)
_iogroup.add_argument(
    "--folder",
    "-f",
    dest="input_dir",
    action="store",
    type=Path,
    default=Path(),
    help=(
        "OPTIONAL: supply the path to the folder where the .D files are stored."
        " If -f is not supplied, then the path is assumed to be the current folder."
    ),
)
_iogroup.add_argument(
    "--out-folder",
    "-o",
    dest="output_dir",
    action="store",
    type=Path,
    default=None,
    help=(
        "OPTIONAL: supply the path to the folder where the vtu outputs should be saved to."
        " Output folder fallback is .output_dir > .input_dir > current working directory."
    ),
)
_iogroup.add_argument(
    "-fo", "-of", dest="input_dir", action=SetBothAction, type=Path, help=argparse.SUPPRESS
)
