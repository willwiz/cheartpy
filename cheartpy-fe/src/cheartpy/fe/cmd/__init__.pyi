from pathlib import Path
from typing import TypedDict, Unpack

class _RunOptions(TypedDict, total=False):
    pedantic: bool
    dump_matrix: bool
    output: bool
    cores: int
    log: Path | str

def run_prep(pfile: Path | str, **kwargs: Unpack[_RunOptions]) -> int: ...
def run_problem(pfile: Path | str, **kwargs: Unpack[_RunOptions]) -> int: ...
