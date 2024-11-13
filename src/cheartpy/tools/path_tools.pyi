__all__ = ["path", "Clear_Dir"]
from .basiclogging import ILogger, NullLogger

def path(*v: str | None) -> str:
    """
    Joins args as a system path str
    """
    ...

def Clear_Dir(folder: str, *suffix: str, LOG: ILogger = NullLogger()) -> None:
    """
    Remove all files in directory
    """
    ...
