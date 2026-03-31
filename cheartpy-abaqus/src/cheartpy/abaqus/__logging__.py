from typing import TYPE_CHECKING, Protocol, Unpack

from cheartpy.io import fix_ch_sfx

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .parsing import AbaqusAPIKwargs


class _Printable(Protocol):
    def __str__(self) -> str: ...


def header_guard(msg: str = "", *, length: int = 80, char: str = "*") -> str:
    return f"{msg:{char}^{length}}"


def compose_header() -> Sequence[str]:
    return [
        header_guard(),
        "    Convert Abaqus Meshes (*.inp) CHeart format.",
        "    This program is part of the CHeart project.",
        "    Author: Andreas Hessenthaler (Original)",
        "            Will Zhang",
        "    Date: 1/20/2026",
        header_guard(),
    ]


def format_boundary_arguments(boundary: Mapping[int, Sequence[str]] | None) -> Sequence[_Printable]:
    if boundary is None:
        return [f"{'<<< Boundary patches are not requested':<40}"]
    return [f"{'<<< The boundary patches are:':<40}", boundary]


def format_input_kwargs(
    files: Sequence[Path | str], **kwargs: Unpack[AbaqusAPIKwargs]
) -> Sequence[_Printable]:
    return [
        f"{'<<< The Abaqus meshes are imported from:'}",
        [str(f) for f in files],
        f"{'<<< The topology are combined form labels:':<40}",
        kwargs["topology"],
        *format_boundary_arguments(kwargs.get("boundary")),
        *(
            ["No prefix is provided: file export not requested"]
            if (prefix := kwargs.get("prefix")) is None
            else [
                f"{'<<< The mesh will be saved to:':<40}",
                *[
                    f"{fix_ch_sfx(prefix)}{ext}"
                    for ext in (("X", "T", "B") if kwargs.get("boundary") else ("X", "T"))
                ],
            ]
        ),
    ]
