__all__ = ["PEXEC_ARGS", "parallel_exec"]
from concurrent import futures
from typing import Any, Callable, Collection, Mapping, Sequence

PEXEC_ARGS = Collection[tuple[Sequence[Any], Mapping[str, Any]]]

def parallel_exec(
    exec: futures.ProcessPoolExecutor,
    func: Callable[..., Any],
    args: Collection[tuple[Sequence[Any], Mapping[str, Any]]],
    prog_bar: bool = False,
) -> None: ...
