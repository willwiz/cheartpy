#!/usr/bin/env python3


from typing import Any, Type
import enum


def VoS(x):
    return str(x) if isinstance(x, (str, int, float)) else x.name


def join_fields(*terms: Any, char: str = "|") -> str:
    vals = [str(v) for v in terms if v is not None]
    return char.join(vals)


def hline(s: str):
    return f"% ----  {s + "  ":-<82}\n"


def cline(s: str):
    return f"% {s}\n"


def header(msg="Begin P file"):
    ls = f"% {'-'*88}\n"
    for s in msg.splitlines():
        ls = ls + cline(s)
    ls = ls + f"% {'-'*88}\n"
    return ls


def splicegen(maxchars, stringlist):
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


class MissingArgument(Exception):
    """Raised when arguments are missing"""

    def __init__(self, message="Missing arguments"):
        self.message = message
        super().__init__(self.message)


def get_enum[T:enum.Enum](v: str | T, e: Type[T]) -> T:
    if not isinstance(v, str):
        return v
    return e[v]
