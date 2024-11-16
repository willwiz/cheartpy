import os
from glob import glob
from .basiclogging import ILogger, NullLogger


def path(*v: str | None) -> str:
    """
    Joins args as a system path str
    """
    return os.path.join(*[s for s in v if s])


def Clear_Dir(folder: str, *suffix: str, LOG: ILogger = NullLogger()) -> None:
    """
    Remove all files in directory
    """
    if not os.path.isdir(folder):
        LOG.warn(f"Dir {folder} was not found.")
        os.makedirs(folder, exist_ok=True)
    if len(suffix) == 0:
        [os.remove(v) for v in glob(f"{folder}/*") if os.path.isfile(v)]
        return
    for s in suffix:
        [os.remove(v) for v in glob(f"{folder}/*.{s}") if os.path.isfile(v)]
