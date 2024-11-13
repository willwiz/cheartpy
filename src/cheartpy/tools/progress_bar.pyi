__all__ = ["ProgressBar"]
from typing import Literal

class ProgressBar:
    def __init__(
        self,
        max: int,
        prefix: str = "",
        suffix: str = "",
        length: int = 50,
        decimal: int = 1,
        pixel: str = "*",
        end: Literal["\n", "\r", ""] = "",
    ) -> None: ...
    def reset(self) -> None: ...
    def next(self) -> None: ...
    def finish(self) -> None: ...
    def _print_bar(self, i: int) -> None: ...
