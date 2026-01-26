from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.io.api import chread_b_utf
from cheartpy.vtk.api import get_vtk_elem
from cheartpy.xml import XMLElement
from pytools.logging import NLOGGER, ILogger
from pytools.parallel import ThreadedRunner, ThreadMethods
from pytools.progress import ProgressBar

from ._caching import get_arguments, get_variables
from ._third_party import compress_vtu

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.search.trait import IIndexIterator
    from cheartpy.vtk.types import VtkType
    from pytools.arrays import A1, A2

    from ._struct import ParaviewTopology, ProgramArgs, VariableCache, XMLDataInputs

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
    top: ParaviewTopology[F, I],
    log: ILogger,
) -> None:
    log.debug("<<< Working on", inp.bfile)
    if inp.bfile is None or top.vtksurfacetype is None:
        log.info(">>> NOTICE: No boundary file given, export is skipped")
        return
    raw = chread_b_utf(inp.bfile)
    db = raw[:, 1:-1] - 1
    dbid = raw[:, -1]
    vtk_xml = create_xml_for_boundary(inp.prefix, top.x, top.vtksurfacetype, db, dbid)
    foutfile = inp.output_dir / f"{inp.prefix}_boundary.vtu"
    with Path(foutfile).open("w") as fout:
        vtk_xml.write(fout)
    if inp.compress:
        compress_vtu(foutfile, log=log)
    log.info(f"<<< Exported the boundary to {foutfile}")


def create_xml_for_mesh[F: np.floating, I: np.integer](
    prefix: str,
    top: ParaviewTopology[F, I],
    x: A2[F],
    var: Mapping[str, A2[F]],
) -> XMLElement:
    vtkfile = XMLElement("VTKFile", type="UnstructuredGrid")
    grid = vtkfile.create_elem(XMLElement("UnstructuredGrid"))
    piece = grid.create_elem(
        XMLElement(
            "Piece",
            Name=f"{prefix}",
            NumberOfPoints=f"{len(x)}",
            NumberOfCells=f"{top.ne}",
        ),
    )
    points = piece.create_elem(XMLElement("Points"))
    dataarr = points.create_elem(
        XMLElement("DataArray", type="Float64", NumberOfComponents="3", Format="ascii"),
    )
    dataarr.add_data(x)

    cell = piece.create_elem(XMLElement("Cells"))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii"),
    )
    dataarr.add_data(top.t, order=get_vtk_elem(top.vtkelementtype).connectivity)
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii"),
    )
    dataarr.add_data(np.arange(top.nc, top.nc * (top.ne + 1), top.nc))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int8", Name="types", Format="ascii"),
    )
    dataarr.add_data(np.full((top.ne,), top.vtkelementtype.value.idx))
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
    args: XMLDataInputs[F, I],
    log: ILogger,
) -> None:
    x, var = get_variables(args.top, args.x, args.u, args.var)
    vtk_xml = create_xml_for_mesh(args.prefix, args.top, x, var)
    with args.path.open("w") as fout:
        vtk_xml.write(fout)
    if args.compress:
        compress_vtu(args.path, log=log)


def run_exports_in_series[F: np.floating, I: np.integer](
    inp: ProgramArgs,
    indexer: IIndexIterator,
    cache: VariableCache[F, I],
    log: ILogger,
) -> None:
    bart = ProgressBar(len(indexer)) if inp.prog_bar else None
    for arg in get_arguments(inp, cache, indexer, log=log):
        log.debug("<<< Working on", arg.path.name)
        export_mesh_iter(arg, log)
        bart.next() if bart else print(f"<<< Completed {arg.path}")


def run_exports_in_parallel[F: np.floating, I: np.integer](
    mpi: ThreadMethods,
    inp: ProgramArgs,
    indexer: IIndexIterator,
    cache: VariableCache[F, I],
    log: ILogger,
) -> None:
    bart = ProgressBar(len(indexer)) if inp.prog_bar else None
    with ThreadedRunner(**mpi, prog_bar=bart) as executor:
        for arg in get_arguments(inp, cache, indexer, log=log):
            executor.submit(export_mesh_iter, arg, log=NLOGGER)
