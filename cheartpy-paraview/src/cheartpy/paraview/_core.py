from concurrent import futures
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.io.api import chread_b_utf
from cheartpy.vtk.api import get_vtk_elem
from cheartpy.xml import XMLElement
from pytools.logging import NLOGGER, ILogger
from pytools.parallel import PEXEC_ARGS, parallel_exec
from pytools.progress import ProgressBar

from ._caching import update_variable_cache
from ._third_party import compress_vtu
from ._variable_getter import CheartVTUFormat

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.search.trait import IIndexIterator
    from cheartpy.vtk.types import VtkType
    from pytools.arrays import A1, A2

    from ._struct import ParaviewTopology, ProgramArgs, VariableCache

__all__ = [
    "export_boundary",
    "run_exports_in_parallel",
    "run_exports_in_series",
]


def create_xml_for_boundary[I: np.integer, F: np.floating](
    prefix: str,
    fx: A2[F],
    vtk_id: VtkType,
    fb: A2[I],
    fbid: A1[I],
) -> XMLElement:
    vtkfile = XMLElement("VTKFile", type="UnstructuredGrid")
    grid = vtkfile.create_elem(XMLElement("UnstructuredGrid"))
    piece = grid.create_elem(
        XMLElement(
            "Piece",
            Name=f"{prefix}",
            NumberOfPoints=f"{len(fx)}",
            NumberOfCells=f"{fb.shape[0]}",
        ),
    )
    dataarr = piece.create_elem(XMLElement("Points")).create_elem(
        XMLElement("DataArray", type="Float64", NumberOfComponents="3", Format="ascii"),
    )
    dataarr.add_data(fx)
    cell = piece.create_elem(XMLElement("CellData", Scalars="scalars"))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int8", Name="PatchIDs", Format="ascii"),
    )
    dataarr.add_data(fbid)
    cell = piece.create_elem(XMLElement("Cells", Scalars="scalars"))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii"),
    )
    dataarr.add_data(fb, order=get_vtk_elem(vtk_id).connectivity)
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii"),
    )
    dataarr.add_data(np.arange(fb.shape[1], fb.size + 1, fb.shape[1]))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int8", Name="types", Format="ascii"),
    )
    dataarr.add_data(np.full((fb.shape[0],), vtk_id.value.idx))
    return vtkfile


def export_boundary[F: np.floating, I: np.integer](
    inp: ProgramArgs,
    cache: VariableCache[F, I],
    log: ILogger,
) -> None:
    log.debug("<<< Working on", inp.bfile)
    if inp.bfile is None:
        log.info(">>> NOTICE: No boundary file given, export is skipped")
        return
    dx = cache.space
    raw = chread_b_utf(inp.bfile)
    db = raw[:, 1:-1] - 1
    dbid = raw[:, -1]
    if cache.top.vtksurfacetype is None:
        log.error(">>> Boundary file does not have a valid surface type")
        return
    vtk_xml = create_xml_for_boundary(inp.prefix, dx, cache.top.vtksurfacetype, db, dbid)
    foutfile = inp.output_dir / f"{inp.prefix}_boundary.vtu"
    with Path(foutfile).open("w") as fout:
        vtk_xml.write(fout)
    if inp.compress:
        compress_vtu(foutfile, log=log)
    log.info(f"<<< Exported the boundary to {foutfile}")


def create_xml_for_mesh[F: np.floating, I: np.integer](
    prefix: str,
    tp: ParaviewTopology[I],
    fx: A2[F],
    var: Mapping[str, A2[F]],
) -> XMLElement:
    vtkfile = XMLElement("VTKFile", type="UnstructuredGrid")
    grid = vtkfile.create_elem(XMLElement("UnstructuredGrid"))
    piece = grid.create_elem(
        XMLElement(
            "Piece",
            Name=f"{prefix}",
            NumberOfPoints=f"{fx.shape[0]}",
            NumberOfCells=f"{tp.ne}",
        ),
    )
    points = piece.create_elem(XMLElement("Points"))
    dataarr = points.create_elem(
        XMLElement("DataArray", type="Float64", NumberOfComponents="3", Format="ascii"),
    )
    dataarr.add_data(fx)

    cell = piece.create_elem(XMLElement("Cells"))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii"),
    )
    dataarr.add_data(tp.get_data(), order=get_vtk_elem(tp.vtkelementtype).connectivity)
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii"),
    )
    dataarr.add_data(np.arange(tp.nc, tp.nc * (tp.ne + 1), tp.nc))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int8", Name="types", Format="ascii"),
    )
    dataarr.add_data(np.full((tp.ne,), tp.vtkelementtype.value.idx))
    points = piece.create_elem(XMLElement("PointData", Scalars="scalars"))
    for v, dv in var.items():
        dataarr = points.create_elem(
            XMLElement(
                "DataArray",
                type="Float64",
                Name=f"{v}",
                NumberOfComponents=f"{dv.shape[1]}",
                Format="ascii",
            ),
        )
        dataarr.add_data(dv)
    return vtkfile


def export_mesh_iter[F: np.floating, I: np.integer](
    prefix: Path,
    t: str | int,
    inp: ProgramArgs,
    cache: VariableCache[F, I],
    log: ILogger,
) -> None:
    log.debug("<<< Working on", prefix.name)
    x, var = update_variable_cache(inp, t, cache, log)
    log.debug("<<< showing cache")
    log.debug(cache)
    vtk_xml = create_xml_for_mesh(inp.prefix, cache.top, x, var)
    with prefix.open("w") as fout:
        vtk_xml.write(fout)
    if inp.compress:
        compress_vtu(prefix, log=log)


def run_exports_in_series[F: np.floating, I: np.integer](
    inp: ProgramArgs,
    indexer: IIndexIterator,
    cache: VariableCache[F, I],
    log: ILogger,
) -> None:
    name = CheartVTUFormat(inp.output_dir, inp.prefix)
    bart = ProgressBar(len(indexer)) if inp.prog_bar else None
    for t in indexer:
        export_mesh_iter(name[t], t, inp, cache, log)
        bart.next() if bart else print(f"<<< Completed {name[t]}")


def run_exports_in_parallel[F: np.floating, I: np.integer](
    inp: ProgramArgs,
    indexer: IIndexIterator,
    cache: VariableCache[F, I],
) -> None:
    name = CheartVTUFormat(inp.output_dir, inp.prefix)
    args: PEXEC_ARGS = [([name[t], t, inp, cache, NLOGGER], {}) for t in indexer]
    bart = ProgressBar(len(indexer)) if inp.prog_bar else None
    with futures.ProcessPoolExecutor(inp.cores) as executor:
        parallel_exec(executor, export_mesh_iter, args, prog_bar=bart)
