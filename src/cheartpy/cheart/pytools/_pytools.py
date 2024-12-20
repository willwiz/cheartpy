__all__ = ["join_fields", "hline", "cline", "header", "splicegen", "get_enum"]
from typing import Any, Type
import enum


def join_fields(*terms: Any, char: str = "|") -> str:
    vals = [
        f"{v[0]}.{v[1]}" if isinstance(v, tuple) else str(v)
        for v in terms
        if v is not None
    ]
    return char.join(vals)


def hline(s: str):
    return f"% ----  {s + "  ":-<82}\n"


def cline(s: str):
    return f"% {s}\n"


def header(msg: str = "Begin P file"):
    ls = f"% {'-'*88}\n"
    for s in msg.splitlines():
        ls = ls + cline(s)
    ls = ls + f"% {'-'*88}\n"
    return ls


def splicegen(maxchars: int, stringlist: list[str]):
    """
    Return a list of slices to print based on maxchars string-length boundary.
    """
    runningcount = 0  # start at 0
    tmpslice = []  # tmp list where we append slice numbers.
    for item in stringlist:
        runningcount += len(item)
        if runningcount <= int(maxchars):
            tmpslice.append(item)
        else:
            yield tmpslice
            tmpslice = [item]
            runningcount = len(item)
    yield (tmpslice)


def get_enum[T: enum.Enum](v: str | T, e: Type[T]) -> T:
    if not isinstance(v, str):
        return v
    return e[v]
