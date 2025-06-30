__all__ = ["cline", "get_enum", "header", "hline", "join_fields", "splicegen"]
import enum
from collections.abc import Generator


def join_fields(*terms: object, char: str = "|") -> str:
    vals = [f"{v[0]}.{v[1]}" if isinstance(v, tuple) else str(v) for v in terms if v is not None]
    return char.join(vals)


def hline(s: str) -> str:
    return f"% ----  {s + '  ':-<82}\n"


def cline(s: str) -> str:
    return f"% {s}\n"


def header(msg: str = "Begin P file") -> str:
    ls = f"% {'-' * 88}\n"
    for s in msg.splitlines():
        ls = ls + cline(s)
    return ls + f"% {'-' * 88}\n"


def splicegen(maxchars: int, stringlist: list[str]) -> Generator[list[str], None, None]:
    """Return a list of slices to print based on maxchars string-length boundary."""
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


def get_enum[T: enum.Enum](v: str | T, e: type[T]) -> T:
    if not isinstance(v, str):
        return v
    return e[v]
