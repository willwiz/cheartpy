from typing import Callable, Literal
import re
import json
import numpy as np
from ...cheart_mesh.io import *
from ..core.element_mapping import (
    ElementTypes,
    get_elem_type,
    get_elmap,
)
from ...cheart2vtu_core.file_indexing import (
    DFileAutoFinder,
    DFileAutoSubIndex,
    DFileIndex,
    DFileSubIndex,
)

from ...var_types import *


def string_head(str1: str, str2: str) -> str:
    n = min(len(str1), len(str2))
    for i in range(n):
        if str1[i] != str2[i]:
            return str1[:i]
    return str1[:n]


def split_Dfile_name(name: str, suffix: str):
    matched = re.search(rf"(^.*)-(\d+).(D|D.gz)", name)
    if matched is None:
        return name + suffix
    print(matched.groups())
    prefix, num, _ = matched.groups()
    return f"{prefix}{suffix}-{num}.D"


def load_topologies(lin: str, quad: str, kind: Literal["TET", "HEX", "SQUARE"] | None):
    lin_top = CHRead_t_utf(lin)
    quad_top = CHRead_t_utf(quad)
    lin_ne, _ = CHRead_header_utf(lin)
    quad_ne, _ = CHRead_header_utf(quad)
    if lin_ne != quad_ne:
        raise ValueError(
            f"Lin top ne = {lin_ne} does not match quad top ne = {quad_ne}"
        )
    if lin_top.shape[0] != lin_ne:
        raise ValueError(
            f"Lin top ne = {lin_top.shape[0]} does not match header = {lin_ne}"
        )
    if quad_top.shape[0] != quad_ne:
        raise ValueError(
            f"Lin top ne = {quad_top.shape[0]} does not match header = {quad_ne}"
        )
    if lin_top.shape[0] == quad_top.shape[0]:
        ValueError("Topologies do not have the same number of elements")
    if kind is None:
        elem = get_elem_type(lin_top.shape[1], quad_top.shape[1])
    else:
        elem = ElementTypes[kind]
    elmap = get_elmap(elem, lin_top.shape[1], quad_top.shape[1])
    return lin_top - 1, quad_top - 1, elmap


def gen_map(
    lin: str,
    quad: str,
    kind: Literal["TET", "HEX", "SQUARE"] | None,
    bar: Callable | None = None,
):
    map: dict[int, list[int]] = dict()
    lin_top, quad_top, elmap = load_topologies(lin, quad, kind)
    rows_quad, _ = quad_top.shape
    for i in range(rows_quad):
        for j, e in enumerate(elmap):
            if quad_top[i, j] not in map:
                map[int(quad_top[i, j])] = [int(lin_top[i, k]) for k in e]
        if bar:
            bar()
    return map


def keystoint(x):
    return {int(k): v for k, v in x.items()}


def load_map(name: str) -> dict[int, list[int]]:
    with open(name, "r") as f:
        map = json.load(f, object_hook=keystoint)
    return map


def interp_lin_to_quad(map: dict[int, list[int]], lin: Mat[f64]):
    quad_data = np.zeros((len(map), lin.shape[1]), dtype=np.float64)
    for k, v in map.items():
        quad_data[k] = lin[v].mean(axis=0)
    return quad_data


def get_file_name_indexer(
    vars: list[str],
    index: tuple[int, int, int] | None,
    sub_auto: bool,
    sub_index: tuple[int, int, int] | None,
    input_folder: str = "",
):
    match index, sub_auto, sub_index:
        case None, _, None:
            print("<<< Acquring DFileAutoFinder")
            return DFileAutoFinder(input_folder, vars, None, sub_auto)
        case (int(), int(), int()), False, None:
            print("<<< Acquring DFileIndex")
            return DFileIndex(input_folder, vars, index)
        case (int(), int(), int()), True, None:
            print("<<< Acquring DFileAutoSubIndex")
            return DFileAutoSubIndex(input_folder, vars, index)
        case (int(), int(), int()), False, (int(), int(), int()):
            print("<<< Acquring DFileSubIndex")
            return DFileSubIndex(input_folder, vars, index, sub_index)
        case _:
            raise ValueError(
                f"Option with index={index}, sub_index={
                    sub_index}, sub_auto={sub_auto} is not recognized"
            )
