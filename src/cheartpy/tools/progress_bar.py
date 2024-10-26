from typing import Final, Literal


class ProgressBar:
    __slots__ = ["_n", "i", "_l", "_p", "_s", "_b", "_bmt", "_pmt", "_endl"]

    _n: Final[int]
    i: int
    _l: Final[int]
    _p: Final[str]
    _s: Final[str]
    _b: Final[str]
    _endl: Final[Literal["\n", "\r", ""]]
    _bmt: Final[str]
    _pmt: Final[str]

    def __init__(
        self,
        max: int,
        prefix: str = "",
        suffix: str = "",
        length: int = 50,
        decimal: int = 1,
        pixel: str = "*",
        end: Literal["\n", "\r", ""] = "",
    ):
        self.i, self._n = 0, max
        self._p, self._s = prefix, suffix
        self._l = length * len(pixel)
        self._b = pixel
        self._bmt = f"-<{self._l}"
        self._pmt = f">{5+decimal}.{decimal}%"
        self._endl = end

    def reset(self) -> None:
        self.i = 0

    def next(self):
        self.i = self.i + 1
        self._print_bar(self.i)

    def finish(self):
        self._print_bar(self._n)

    def _print_bar(self, i: int):
        p = 1.0 if (self._n == 0) else (i / self._n)
        f = self._b * int(self._l * p)
        bar = f"\r{self._p}|{f:{self._bmt}}|{p:{self._pmt}}{self._s}"
        print(bar, end=self._endl)
        if i == self._n:
            print()
