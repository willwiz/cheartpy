__all__ = [
    "ZeroIndexer",
    "RangeIndexer",
    "RangeSubIndexer",
    "ListIndexer",
    "ListSubIndexer",
    "TupleIndexer",
]
from typing import Final, Iterator, Mapping
from .interfaces import *


class ZeroIndexer(IIndexIterator):
    __slots__ = ["size"]
    size: Final[int]

    def __init__(self) -> None:
        self.size = 1

    def __iter__(self) -> Iterator[int]:
        yield 0

    def __len__(self) -> int:
        return 1

    @property
    def mode(self):
        return ProgramMode.none


class RangeIndexer(IIndexIterator):
    __slots__ = ["i0", "it", "di", "size"]
    i0: Final[int]
    it: Final[int]
    di: Final[int]
    size: Final[int]

    def __init__(self, index: tuple[int, int, int]) -> None:
        self.i0 = index[0]
        self.it = index[1]
        self.di = index[2]
        self.size = (index[1] - index[0]) // index[2] + 1

    def __iter__(self) -> Iterator[int]:
        for i in range(self.i0, self.it, self.di):
            yield i

    def __len__(self) -> int:
        return self.size

    @property
    def mode(self):
        return ProgramMode.range


class RangeSubIndexer(IIndexIterator):
    __slots__ = ["i0", "it", "di", "s0", "st", "ds", "size"]
    i0: Final[int]
    it: Final[int]
    di: Final[int]
    s0: Final[int]
    st: Final[int]
    ds: Final[int]
    size: Final[int]

    def __init__(
        self,
        index: tuple[int, int, int],
        sub_index: tuple[int, int, int],
    ) -> None:
        self.i0 = index[0]
        self.it = index[1]
        self.di = index[2]
        self.s0 = sub_index[0]
        self.st = sub_index[1]
        self.ds = sub_index[2]
        self.size = ((index[1] - index[0]) // index[2] + 1) * (
            (sub_index[1] - sub_index[0]) // sub_index[2] + 2
        )

    def __iter__(self) -> Iterator[str]:
        for i in range(self.i0, self.it, self.di):
            yield f"{i}"
            for j in range(self.s0, self.st, self.ds):
                yield f"{i}.{j}"

    def __len__(self) -> int:
        return self.size

    @property
    def mode(self):
        return ProgramMode.subindex


class ListIndexer[T: (int, str)](IIndexIterator):
    __slots__ = ["indices"]
    values: list[T]

    def __init__(self, values: list[T]) -> None:
        self.values = values

    def __iter__(self) -> Iterator[T]:
        for i in self.values:
            yield i

    def __len__(self) -> int:
        return len(self.values)

    @property
    def mode(self):
        return ProgramMode.search


class ListSubIndexer(IIndexIterator):
    __slots__ = ["values", "si"]
    values: list[int]
    si: tuple[int, int, int]

    def __init__(self, indices: list[int], sub_index: tuple[int, int, int]) -> None:
        self.values = indices
        self.si = sub_index

    def __iter__(self) -> Iterator[str]:
        for i in self.values:
            yield str(i)
            for j in range(*self.si):
                yield f"{i}.{j}"

    def __len__(self) -> int:
        return len(self.values) * ((self.si[1] - self.si[0]) // self.si[2] + 2)

    @property
    def mode(self):
        return ProgramMode.searchsubindex


class TupleIndexer(IIndexIterator):
    __slots__ = ["values"]
    values: Mapping[int, list[int]]

    def __init__(self, values: dict[int, list[int]]) -> None:
        self.values = values

    def __iter__(self) -> Iterator[str]:
        for k, vlist in self.values.items():
            yield str(k)
            for v in vlist:
                yield f"{k}.{v}"

    def __len__(self) -> int:
        return sum([len(v) for v in self.values.values()])

    @property
    def mode(self):
        return ProgramMode.subauto
