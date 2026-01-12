from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TextIO

import numpy as np
from pytools.result import Err, Ok, all_ok

from ._struct import AbaqusContent, AbaqusMeshTuple, MeshElements, MeshNodes
from ._trait import AbaqusItem

if TYPE_CHECKING:
    from pytools.logging.trait import ILogger


class _ElementHeader(NamedTuple):
    setname: str
    settype: str


def read_elements_header(line: str, log: ILogger) -> Ok[_ElementHeader] | Err:
    values = line.strip().split(",")
    setname = None
    settype = None
    for v in values:
        lvalue, rvalue = v.split("=")
        if lvalue.strip().lower() == "type":
            settype = rvalue.strip()
        elif lvalue.strip().lower() == "name":
            setname = rvalue.strip()
        log.debug(f"Found element header: {lvalue.strip()} = {rvalue.strip()}")
    if setname is None:
        msg = "Element header does not contain 'elset' field."
        return Err(ValueError(msg))
    if settype is None:
        msg = "Element header does not contain 'type' field."
        return Err(ValueError(msg))
    return Ok(_ElementHeader(setname, settype))


def get_next_content(line: str, log: ILogger) -> AbaqusContent | None:
    line = line.strip().lower()
    if line.startswith(AbaqusItem.HEADING.value):
        return AbaqusContent(AbaqusItem.HEADING)
    if line.startswith(AbaqusItem.NODES.value):
        return AbaqusContent(AbaqusItem.NODES)
    if line.startswith(AbaqusItem.ELEMENTS.value):
        match read_elements_header(line, log):
            case Ok((name, kind)):
                return AbaqusContent(AbaqusItem.ELEMENTS, (name, kind))
            case Err(e):
                raise e
    if line.startswith(AbaqusItem.COMMENTS.value):
        return AbaqusContent(AbaqusItem.COMMENTS)
    return None


def get_first_content(f: TextIO, log: ILogger) -> Ok[AbaqusContent] | Err:
    for line in f:
        content = get_next_content(line, log)
        if content is not None:
            return Ok(content)
        log.debug(f"Skipping line: {line}")
    msg = "No valid Abaqus mesh content found in the file."
    return Err(ValueError(msg))


def read_comments(f: TextIO, log: ILogger) -> AbaqusContent | None:
    """Return the next Abaqus mesh content from the file if found else None if at  end of file.

    This function will print the headings of the Abaqus mesh file
    Args:
        f (TextIO): A file-like object to read lines from.
        log (ILogger): Logger to log messages.

    Returns:
        AbaqusMeshContent | None: The extracted mesh content if found, otherwise None.

    """
    for line in f:
        content = get_next_content(line, log)
        if content is not None:
            return content
        log.disp(line)
    return None


def read_nodes[F: np.floating](
    f: TextIO,
    log: ILogger,
    *,
    dtype: type[F] = np.float64,
) -> tuple[MeshNodes[F], AbaqusContent | None]:
    """Read nodes from the Abaqus mesh file.

    Args:
        f (TextIO): A file-like object to read lines from.
        log (ILogger): Logger to log messages.
        dtype (type[F], optional): Data type for the node coordinates. Defaults to np.float64.

    Returns:
        list[tuple[int, float, float, float]]: List of nodes with their coordinates.

    """
    log.info("Reading nodes from Abaqus mesh file.")
    nodes: list[list[float]] = []
    for line in f:
        content = get_next_content(line, log)
        if content is not None:
            return MeshNodes(len(nodes), np.array(nodes, dtype=dtype)), content
        nodes.append([float(i) for i in line.strip().split(",")])
    return MeshNodes(len(nodes), np.array(nodes, dtype=dtype)), None


def read_elements[I: np.integer](
    elem: tuple[str, str],
    f: TextIO,
    log: ILogger,
    *,
    dtype: type[I] = np.intc,
) -> tuple[MeshElements[I], AbaqusContent | None]:
    """Read elements from the Abaqus mesh file.

    Args:
        elem (tuple[str, str]): Tuple containing element set name and type.
        f (TextIO): A file-like object to read lines from.
        log (ILogger): Logger to log messages.
        dtype (type[I], optional): Data type for the element indices. Defaults to np.intc.

    Returns:
        MeshElements[I], AbaqusMeshContent | None: A tuple containing the MeshElements object
        and the next content type.

    """
    name, kind = elem
    log.info(f"Reading elements {name}, type {kind} from Abaqus mesh file.")
    elems: list[list[int]] = []
    for line in f:
        content = get_next_content(line, log)
        if content is not None:
            return MeshElements(name, kind, len(elems), np.array(elems, dtype=dtype)), content
        elems.append([int(i) for i in line.strip().split(",")])
    return MeshElements(name, kind, len(elems), np.array(elems, dtype=dtype)), None


class _RawAbaqusMeshTuple[F: np.floating, I: np.integer](NamedTuple):
    nodes: MeshNodes[F] | None
    elements: dict[str, MeshElements[I]]


def abaqus_importer[F: np.floating, I: np.integer](
    file: Path,
    log: ILogger,
    *,
    ftype: type[F] = np.float64,
    itype: type[I] = np.intc,
) -> Ok[_RawAbaqusMeshTuple[F, I]] | Err:
    """Import Abaqus mesh from a file-like object.

    Args:
        file (Path): Path to the Abaqus mesh file.
        log (ILogger): Logger to log messages.
        ftype (type[F], optional): Data type for the node coordinates. Defaults to np.float64.
        itype (type[I], optional): Data type for the element indices. Defaults to np.intc.

    Returns:
        tuple[MeshNodes[np.float64], list[MeshElements[np.intc]]]: A tuple containing the MeshNodes
        and a list of MeshElements objects.

    """
    with file.open("r") as f:
        match get_first_content(f, log):
            case Ok(content):
                nodes: MeshNodes[F] | None = None
                elements: dict[str, MeshElements[I]] = {}
            case Err(e):
                return Err(e)

        while content is not None:
            match content:
                case AbaqusContent(key=AbaqusItem.HEADING):
                    content = read_comments(f, log)
                case AbaqusContent(key=AbaqusItem.COMMENTS):
                    content = read_comments(f, log)
                case AbaqusContent(key=AbaqusItem.ELEMENTS, value=(name, kind)):
                    elem, content = read_elements((name, kind), f, log, dtype=itype)
                    elements[name] = elem
                case AbaqusContent(key=AbaqusItem.NODES) if nodes is None:
                    nodes, content = read_nodes(f, log, dtype=ftype)
                case AbaqusContent(key=AbaqusItem.NODES):
                    msg = "Duplicate nodes section found in the Abaqus mesh file."
                    return Err(ValueError(msg))
                case _:
                    msg = f"Unexpected content type: {content}. Probably implementation error."
                    raise NotImplementedError(msg)
    return Ok(_RawAbaqusMeshTuple(nodes, elements))


def merge_abaqus_meshes[F: np.floating, I: np.integer](
    *meshes: tuple[MeshNodes[F] | None, dict[str, MeshElements[I]]],
) -> Ok[AbaqusMeshTuple[F, I]] | Err:
    nodes = [m[0] for m in meshes if m[0] is not None]
    elems = [m[1] for m in meshes if m[1]]
    dimensions = {n.v.shape[1] for n in nodes}
    if len(dimensions) != 1:
        msg = (
            "All meshes must have the same number of dimensions."
            f"Instead dimensions found were: {dimensions}"
        )
        return Err(ValueError(msg))
    space = MeshNodes(
        n=sum(n.n for n in nodes),
        v=np.concatenate([n.v for n in nodes], axis=0).astype(nodes[0].v.dtype, copy=False),
    )
    elements: dict[str, MeshElements[I]] = {}
    for mesh in elems:
        for name, elem in mesh.items():
            if name not in elements:
                elements[name] = elem
            elif elements[name] != elem:
                msg = (
                    f"Element '{name}' is defined multiple times but is not the same: "
                    f"{elements[name].kind} and {elem.kind}."
                    f"{elements[name].v.shape} vs {elem.v.shape}"
                    f"Please check the Abaqus mesh files for consistency."
                )
                return Err(ValueError(msg))
    return Ok(AbaqusMeshTuple(space, elements))


def read_abaqus_meshes[F: np.floating, I: np.integer](
    *files: str,
    log: ILogger,
    ftype: type[F] = np.float64,
    itype: type[I] = np.intc,
) -> Ok[AbaqusMeshTuple[F, I]] | Err:
    match all_ok([abaqus_importer(Path(f), log, ftype=ftype, itype=itype) for f in files]):
        case Ok(abaqus_meshes):
            return merge_abaqus_meshes(*abaqus_meshes)
        case Err(e):
            return Err(e)
