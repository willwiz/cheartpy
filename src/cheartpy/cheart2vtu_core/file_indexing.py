import os
import re
from glob import glob
from collections import defaultdict
from typing import Final, Generator
import numpy as np
from cheartpy.var_types import i32, char
from cheartpy.cheart2vtu_core.data_types import CmdLineArgs, Arr, ProgramMode


class DFileNoVariable:
    __slots__ = ["size"]
    size: Final[int]

    def __init__(self) -> None:
        self.size = 0

    def get_generator(self) -> Generator[str, None, None]:
        yield "0"


def get_int_from_string_template(template: str, s: str) -> tuple[int, int]:
    matched = re.search(template, os.path.basename(s))
    if matched is None:
        raise ValueError("Unknown Error, regex cannot find int in str variable")
    res = matched.group(1).split(".")
    if len(res) == 1:
        return (int(res[0]), -1)
    elif len(res) == 2:
        return (int(res[0]), int(res[1]))
    else:
        raise ValueError(
            f"Applying template {
                         template}, but index cannot be recognized as int or int.int"
        )


def get_index_from_filenames(
    folder: str, var: str, allow_subindex: bool = False
) -> Arr[tuple[int, int], i32]:
    files = glob(os.path.join(folder, f"{var}-*.D"))
    if not files:
        files = glob(os.path.join(folder, f"{var}-*.D.gz"))
    res = np.array(
        sorted(
            [get_int_from_string_template(rf"{var}-(.+?).(D|D.gz)", s) for s in files]
        ),
        dtype=int,
    )
    if res.size == 0:
        msg = f"No file for {var} found!!"
        print(f">>> ERROR: {msg}")
        raise ValueError(msg)
    if not allow_subindex:
        res = res[res[:, 1] == -1]
    return res


def check_arrays_for_equality(
    array_list: list[Arr[tuple[int, int], i32]]
) -> Arr[tuple[int, int], i32]:
    if len(array_list) == 1:
        return array_list[0]
    for i in range(1, len(array_list)):
        if not np.array_equal(array_list[0], array_list[i]):
            print(
                ">>>WARNING: Not all variables have the same index, find method cannot be used"
            )
    return max(array_list, key=len)


def find_index_from_filenames(
    folder: str, vars: list[str], step: int | None = None, allow_subindex: bool = False
) -> Arr[tuple[int, int], i32]:
    if vars:
        index_from_vars = [
            get_index_from_filenames(folder, v, allow_subindex) for v in vars
        ]
        index = check_arrays_for_equality(index_from_vars)
    else:
        index = np.zeros((0, 2), dtype=np.int32)
    if step is not None:
        index: Arr[tuple[int, int], i32] = index[index[:, 0] % step == 0]
    return index


class DFileAutoFinder:
    __slots__ = ["step", "index", "size"]

    step: Final[int | None]
    index: Final[Arr[int, char]]
    size: Final[int]

    def __init__(
        self,
        folder: str,
        var: list[str],
        step: int | None,
        allow_subindex: bool = False,
    ) -> None:
        self.step = step
        int_index = find_index_from_filenames(folder, var, self.step, allow_subindex)
        self.index = np.array([f"{i}" if j < 0 else f"{i}.{j}" for i, j in int_index])
        self.size = len(self.index)

    def get_generator(self) -> Generator[str, None, None]:
        for i in self.index:
            yield i


def check_variable_exist_by_index(
    folder: str, var: list[str], i0: int, it: int, di: int
):
    for i in range(i0, it, di):
        for v in var:
            if not os.path.isfile(os.path.join(folder, f"{v}-{i}.D")):
                raise ValueError(f"variable {v} cannot be found with index {i}")


class DFileIndex:
    __slots__ = ["i0", "it", "di", "size"]
    i0: Final[int]
    it: Final[int]
    di: Final[int]
    size: Final[int]

    def __init__(
        self, folder: str, var: list[str], index: tuple[int, int, int]
    ) -> None:
        self.i0 = index[0]
        self.it = index[1]
        self.di = index[2]
        check_variable_exist_by_index(folder, var, self.i0, self.it, self.di)
        self.size = len(range(self.i0, self.it, self.di))

    def get_generator(self) -> Generator[str, None, None]:
        for i in range(self.i0, self.it, self.di):
            yield str(i)


def check_variable_exist_by_subindex(
    folder: str, var: list[str], i0: int, it: int, di: int, s0: int, st: int, ds: int
):
    for i in range(i0, it, di):
        for v in var:
            if not os.path.isfile(os.path.join(folder, f"{v}-{i}.D")):
                raise ValueError(f"variable {v} cannot be found with index {i}")
        for j in range(s0, st, ds):
            for v in var:
                if not os.path.isfile(os.path.join(folder, f"{v}-{i}.{j}.D")):
                    print(
                        f">>>WARNING: variable {
                          v} cannot be found with index {i} and subindex {j}"
                    )


class DFileSubIndex:
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
        folder: str,
        var: list[str],
        index: tuple[int, int, int],
        sub_index: tuple[int, int, int],
    ) -> None:
        self.i0 = index[0]
        self.it = index[1]
        self.di = index[2]
        self.s0 = sub_index[0]
        self.st = sub_index[1]
        self.ds = sub_index[2]
        check_variable_exist_by_subindex(
            folder,
            var,
            self.i0,
            self.it,
            self.di,
            self.s0,
            self.st,
            self.ds,
        )
        self.size = len(range(self.i0, self.it, self.di)) * len(
            range(self.s0, self.st, self.ds)
        )

    def get_generator(self) -> Generator[str, None, None]:
        for i in range(self.i0, self.it, self.di):
            yield str(i)
            for j in range(self.s0, self.st, self.ds):
                yield f"{i}.{j}"


def check_variable_exist_by_autosubindex(
    folder: str, var: list[str], i0: int, it: int, di: int
):
    if not var:
        return
    variable_subindex_list = defaultdict(lambda: 0)
    for i in range(i0, it, di):
        variable_subindex_list.clear()
        for v in var:
            if not (
                os.path.isfile(os.path.join(folder, f"{v}-{i}.D"))
                or os.path.isfile(os.path.join(folder, f"{v}-{i}.D.gz"))
            ):
                raise ValueError(f"variable {v} cannot be found with index {i}")
            jlist = [item for item in glob(os.path.join(folder, f"{v}-{i}.D"))]
            if not jlist:
                [item for item in glob(os.path.join(folder, f"{v}-{i}.D.gz"))]
            for j in jlist:
                result = re.search(rf"{v}-(.+?).(D|D.gz)", os.path.basename(j))
                if result is None:
                    print(
                        f"variable {
                          v} cannot cannot be found with subindex {i}.x"
                    )
                else:
                    variable_subindex_list[result.group(1)] += 1
        if len(set(variable_subindex_list.values())) != 1:
            print(">>>WARNING: variable subindex found are not the same")


class DFileAutoSubIndex:
    __slots__ = ["i0", "it", "di", "folder", "var", "size"]
    i0: Final[int]
    it: Final[int]
    di: Final[int]
    folder: Final[str]
    var: Final[str | None]

    def __init__(
        self, folder: str, var: list[str], index: tuple[int, int, int]
    ) -> None:
        self.i0 = index[0]
        self.it = index[1]
        self.di = index[2]
        self.folder = folder
        self.var = var[0] if var else None
        check_variable_exist_by_autosubindex(folder, var, self.i0, self.it, self.di)

    def get_generator(self) -> Generator[str, None, None]:
        if self.var is None:
            return
        for i in range(self.i0, self.it, self.di):
            yield str(i)
            jlist = glob(os.path.join(self.folder, f"{self.var}-{i}.*.D"))
            for j in jlist:
                result = re.search(rf"{self.var}-(.+?).(D|D.gz)", os.path.basename(j))
                if result is not None:
                    yield result.group(1)
                else:
                    raise ValueError("Impossible Error, please report!")


IndexerList = (
    DFileNoVariable | DFileAutoFinder | DFileIndex | DFileSubIndex | DFileAutoSubIndex
)


def get_file_name_indexer(
    args: CmdLineArgs,
) -> tuple[
    ProgramMode,
    DFileNoVariable | DFileAutoFinder | DFileIndex | DFileSubIndex | DFileAutoSubIndex,
]:
    if not args.var:
        print("<<< Variables not given. Exporting mesh only.")
        return (ProgramMode.none, DFileNoVariable())
    index = args.index
    sub_auto = args.sub_auto
    sub_index = args.sub_index
    match index, sub_auto, sub_index:
        case None, _, None if args.cmd == "find":
            print("<<< Acquring DFileAutoFinder")
            return (
                ProgramMode.searchsubindex if args.sub_index else ProgramMode.search,
                DFileAutoFinder(args.input_folder, args.var, args.step, args.sub_auto),
            )
        case (int(), int(), int()), False, None:
            print("<<< Acquring DFileIndex")
            return (
                ProgramMode.range,
                DFileIndex(args.input_folder, args.var, index),
            )
        case (int(), int(), int()), True, None:
            print("<<< Acquring DFileAutoSubIndex")
            return (
                ProgramMode.subauto,
                DFileAutoSubIndex(args.input_folder, args.var, index),
            )
        case (int(), int(), int()), False, (int(), int(), int()):
            print("<<< Acquring DFileSubIndex")
            return (
                ProgramMode.subindex,
                DFileSubIndex(args.input_folder, args.var, index, sub_index),
            )
        case _:
            raise ValueError(
                f"Option with cmd={args.cmd}, index={args.index}, sub_index={
                    args.sub_index}, sub_auto={args.sub_auto} is not recognized"
            )
