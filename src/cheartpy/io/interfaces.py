__all__ = ["IndexIterator"]
import abc
from typing import Iterator


class IndexIterator(abc.ABC):
    @abc.abstractmethod
    def __iter__(self) -> Iterator[str]: ...
