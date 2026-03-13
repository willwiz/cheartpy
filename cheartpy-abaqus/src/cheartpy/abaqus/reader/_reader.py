import re
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

import numpy as np
from cheartpy.elem_interfaces import AbaqusEnum
from pytools.logging import get_logger
from pytools.result import Err, Ok, Result

from ._types import AbaqusHeader, AbaqusMesh, Content, Element, ElSet, Headings, Nodes, NSet
from ._validation import validate_element_dimensions, validate_space_dimension

if TYPE_CHECKING:
    import optype.numpy as opn
    from pytools.arrays import A1, DType


def check_header(raw: str) -> Content | None:
    line = raw.strip().lower()
    for e in AbaqusHeader:
        if line.startswith(e.value):
            return Content(e, raw)
    return None


def read_comments(f: TextIO, first_line: str) -> Ok[tuple[None, Content | None]]:
    log = get_logger()
    comments: list[str] = []
    for line in f:
        match check_header(line):
            case Content() as next_content:
                break
            case None:
                comments.append(line)
    else:
        next_content = None
    log.info(first_line, *comments)
    return Ok((None, next_content))


def read_headings(f: TextIO, _first_line: str) -> Ok[tuple[Headings, Content | None]]:
    headings: list[str] = []
    for line in f:
        match check_header(line):
            case Content() as next_content:
                break
            case None:
                headings.append(line)
    else:
        next_content = None
    return Ok(
        (
            Headings(headings),
            next_content,
        )
    )


def read_nodes[F: np.floating](
    f: TextIO, _first_line: str, *, ftype: DType[F] = np.float64
) -> Ok[tuple[Nodes[F], Content | None]]:
    node_dict: dict[opn.ToInt, A1[F]] = {}
    for line in f:
        match check_header(line):
            case Content() as next_content:
                break
            case None:
                i, *vs = line.strip().split(",")
                node_dict[int(i) - 1] = np.array([float(v) for v in vs], dtype=ftype)
    else:
        next_content = None
    return Ok(
        (
            Nodes(node_dict),
            next_content,
        )
    )


def parse_type_name_from_element_header(line: str) -> Result[tuple[str, AbaqusEnum]]:
    terms = line.strip().split(",")
    match terms:
        case str(), str(kind_str), str(name_str): ...  # fmt: skip
        case _:
            msg = f"Line does not match `*ELEMENT, type={{type}}, ELSET={{name}}\nFound: {line}"
            return Err(ValueError(msg))
    match re.fullmatch(r"ELSET=(.*)", name_str.strip(), re.IGNORECASE):
        case None:
            msg = f"Parsing error for name element: {name_str}, found: `{name_str.strip()}`"
            return Err(ValueError(msg))
        case match_obj:
            name = match_obj.group(1)
            print("Found name =", name)
    match re.fullmatch(r"type=(.*)", kind_str.strip(), re.IGNORECASE):
        case None:
            msg = f"Parsing error for type element: {kind_str}"
            return Err(ValueError(msg))
        case match_obj:
            kind = match_obj.group(1)
    if kind not in AbaqusEnum.__members__:
        msg = f"Element type = {kind} has not been implemented."
        return Err(ValueError(msg))
    return Ok((name, AbaqusEnum[kind]))


def parse_name_from_elset_header(line: str) -> Result[str]:
    terms = line.strip().split(",")
    match terms:
        case str(),str(name_str): ...  # fmt: skip
        case _:
            msg = f"Line does not match `*ELEMENT, ELSET={{name}}\nFound: {line}"
            return Err(ValueError(msg))
    match re.fullmatch(r"ELSET=(.*)", name_str.strip(), re.IGNORECASE):
        case None:
            msg = f"Parsing error for name element: {name_str}"
            return Err(ValueError(msg))
        case match_obj:
            name = match_obj.group(1)
    return Ok(name)


def parse_name_from_nset_header(line: str) -> Result[str]:
    terms = line.strip().split(",")
    match terms:
        case str(),str(name_str): ...  # fmt: skip
        case _:
            msg = f"Line does not match `*ELEMENT, ELSET={{name}}\nFound: {line}"
            return Err(ValueError(msg))
    match re.fullmatch(r"NSET=(\s+)", name_str.strip(), re.IGNORECASE):
        case None:
            msg = f"Parsing error for name element: {name_str}"
            return Err(ValueError(msg))
        case match_obj:
            name = match_obj.group(1)
    return Ok(name)


def read_element[I: np.integer](
    f: TextIO, first_line: str, *, dtype: DType[I] = np.intp
) -> Result[tuple[Element[I], Content | None]]:
    elem_dict: dict[opn.ToInt, A1[I]] = {}
    match parse_type_name_from_element_header(first_line):
        case Ok((name, kind)): ...  # fmt: skip
        case Err(e):
            return Err(e)
    for line in f:
        match check_header(line):
            case Content() as next_content:
                break
            case None:
                i, *vs = line.strip().split(",")
                elem_dict[int(i) - 1] = np.array([int(v) - 1 for v in vs], dtype=dtype)
    else:
        next_content = None
    return Ok(
        (
            Element(name, kind, elem_dict),
            next_content,
        )
    )


def read_nset[I: np.integer](
    f: TextIO, first_line: str, *, dtype: DType[I] = np.intp
) -> Result[tuple[NSet[I], Content | None]]:
    set_ids: list[int] = []
    match parse_name_from_nset_header(first_line):
        case Ok(name): ...  # fmt: skip
        case Err(e):
            return Err(e)
    for line in f:
        match check_header(line):
            case Content() as next_content:
                break
            case None:
                set_ids.extend([int(v.strip()) - 1 for v in line.strip().split(",") if v.strip()])
    else:
        next_content = None
    return Ok(
        (
            NSet(name, np.array(set_ids, dtype=dtype)),
            next_content,
        )
    )


def read_elset[I: np.integer](
    f: TextIO, first_line: str, *, dtype: DType[I] = np.intp
) -> Result[tuple[ElSet[I], Content | None]]:
    set_ids: list[int] = []
    match parse_name_from_elset_header(first_line):
        case Ok(name): ...  # fmt: skip
        case Err(e):
            return Err(e)
    for line in f:
        match check_header(line):
            case Content() as next_content:
                break
            case None:
                set_ids.extend([int(v.strip()) - 1 for v in line.strip().split(",") if v.strip()])
    else:
        next_content = None
    return Ok(
        (
            ElSet(name, np.array(set_ids, dtype=dtype)),
            next_content,
        )
    )


type _AbaqusItem[F: np.floating, I: np.integer] = (
    Headings | Nodes[F] | Element[I] | NSet[I] | ElSet[I] | None
)


def read_next[F: np.floating, I: np.integer](
    content: Content,
    f: TextIO,
    *,
    ftype: DType[F] = np.float64,
    dtype: DType[I] = np.intc,
) -> Result[tuple[_AbaqusItem[F, I], Content | None]]:
    match content.type:
        case AbaqusHeader.COMMENTS:
            return read_comments(f, content.next).next()
        case AbaqusHeader.HEADINGS:
            return read_headings(f, content.next).next()
        case AbaqusHeader.NODES:
            return read_nodes(f, content.next, ftype=ftype).next()
        case AbaqusHeader.ELEMENT:
            return read_element(f, content.next, dtype=dtype).next()
        case AbaqusHeader.NSET:
            return read_nset(f, content.next, dtype=dtype).next()
        case AbaqusHeader.ELSET:
            return read_elset(f, content.next, dtype=dtype).next()


def update_abaqus_mesh[F: np.floating, I: np.integer](
    mesh: AbaqusMesh[F, I], item: _AbaqusItem[F, I]
) -> AbaqusMesh[F, I]:
    match item:
        case Headings():
            return mesh.add_headings(item)
        case Nodes():
            return mesh.add_nodes(item)
        case Element():
            return mesh.add_element(item)
        case NSet():
            return mesh.add_nset(item)
        case ElSet():
            return mesh.add_elset(item)
        case None:
            return mesh


def _import_abaqus_file[F: np.floating, I: np.integer](
    f: TextIO, *, ftype: DType[F] = np.float64, dtype: DType[I] = np.intp
) -> Result[AbaqusMesh[F, I]]:
    first_line = f.readline()
    next_content = check_header(first_line)
    mesh = AbaqusMesh(Headings(v=[]), Nodes({}), {}, {}, {}, ftype, dtype)
    while next_content is not None:
        match read_next(next_content, f, ftype=ftype, dtype=dtype):
            case Ok((new_item, next_content)):
                mesh = update_abaqus_mesh(mesh, new_item)
            case Err(e):
                return Err(e)
    if not validate_space_dimension(mesh):
        msg = "Inconsistent space dimensions among nodes."
        return Err(ValueError(msg))
    if not validate_element_dimensions(mesh):
        msg = "Inconsistent element dimensions among elements."
        return Err(ValueError(msg))
    return Ok(mesh)


def import_abaqus_file[F: np.floating, I: np.integer](
    file_name: Path | str, *, ftype: DType[F] = np.float64, dtype: DType[I] = np.intp
) -> Result[AbaqusMesh[F, I]]:
    file_name = Path(file_name)
    with file_name.open("r") as f:
        return _import_abaqus_file(f, ftype=ftype, dtype=dtype).next()
