from typing import TYPE_CHECKING, Protocol, Unpack

from cheartpy.io.api import fix_ch_sfx

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ._types import AbaqusAPIKwargs


class _Printable(Protocol):
    def __str__(self) -> str: ...


def header_guard(*, length: int = 80, char: str = "*") -> str:
    return char * length


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


def format_input_kwargs(**kwargs: Unpack[AbaqusAPIKwargs]) -> Sequence[_Printable]:
    return [
        f"{'<<< The Abaqus meshes are imported from:'}",
        [str(f) for f in kwargs["files"]],
        f"{'<<< The topology are combined form labels:':<40}",
        kwargs["topology"],
        *format_boundary_arguments(kwargs.get("boundary")),
        f"{'<<< The mesh will be saved to:':<40}",
        *[
            f"{fix_ch_sfx(kwargs['prefix'])}{ext}"
            for ext in (("X", "T", "B") if kwargs.get("boundary") else ("X", "T"))
        ],
    ]
