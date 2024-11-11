__all__ = ["PEXEC_ARGS", "parallel_exec"]
from concurrent import futures
from typing import Any, Callable, Collection
from .progress_bar import ProgressBar

PEXEC_ARGS = Collection[tuple[list[Any], dict[str, Any]]]


def parallel_exec(
    exec: futures.ProcessPoolExecutor,
    func: Callable[..., Any],
    args: Collection[tuple[list[Any], dict[str, Any]]],
    prog_bar: bool = False,
):
    jobs: dict[futures.Future[Any], int] = dict()
    bart = ProgressBar(max=len(args)) if prog_bar else None
    for i, (a, k) in enumerate(args):
        a = exec.submit(func, *a, **k)
        jobs[a] = i
    for future in futures.as_completed(jobs):
        try:
            future.result()
            bart.next() if bart else print(f"<<< Completed {jobs[future]}")
        except Exception as e:
            raise e
