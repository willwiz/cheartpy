import dataclasses as dc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


@dc.dataclass(slots=True)
class ParsedInput:
    files: Sequence[str]
    prefix: str | None
    topology: Sequence[str]
    boundary: Sequence[str] | None
    add_mask: Sequence[str] | None
    cores: int
