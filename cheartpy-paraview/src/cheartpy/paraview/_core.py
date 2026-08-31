from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from cheartpy.io import chread_b_utf
from cheartpy.vtk.api import get_vtk_elem
from cheartpy.xml import XMLElement
from pytools.logging import ILogger, get_logger
from pytools.parallel import ThreadedRunner, ThreadMethods
from pytools.progress import ProgressBar

from ._caching import get_arguments, get_xml_variables
from ._third_party import compress_vtu

if TYPE_CHECKING:
    from collections.abc import Mapping

    from cheartpy.elem_interfaces import VtkEnum
    from cheartpy.search import IIndexIterator
    from pytools.arrays import A1, A2

    from ._struct import ParaviewTopology, ProgramArgs, VariableCache, XMLDataInputs

__all__ = [
    "export_boundary",
    "run_exports_in_parallel",
    "run_exports_in_series",
]

_2D = 2


def convert_3d[F: np.floating](arr: A2[F]) -> A2[F]:
    if arr.shape[1] == _2D:
        return np.hstack((arr, np.zeros((arr.shape[0], 1), dtype=arr.dtype)))
    return arr


def create_xml_for_boundary[I: np.integer, F: np.floating](
    prefix: str,
    fx: A2[F],
    vtk_id: VtkEnum,
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
    fx = convert_3d(fx)
    dataarr.add_data(fx)
    cell = piece.create_elem(XMLElement("CellData", Scalars="scalars"))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int8", Name="PatchIDs", Format="ascii"),
    )
    dataarr.add_data(fbid.astype(np.int8))
    cell = piece.create_elem(XMLElement("Cells", Scalars="scalars"))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii"),
    )
    dataarr.add_data(fb.astype(np.int64), order=get_vtk_elem(vtk_id).connectivity)
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii"),
    )
    dataarr.add_data(np.arange(fb.shape[1], fb.size + 1, fb.shape[1], dtype=np.int64))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int8", Name="types", Format="ascii"),
    )
    dataarr.add_data(np.full((fb.shape[0],), vtk_id.value.idx, dtype=np.int8))
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
    log.info("Exported the boundary to:", f"{foutfile!s}")


def create_xml_for_mesh[F: np.floating, I: np.integer](
    prefix: str,
    top: ParaviewTopology[F, I],
    x: A2[F],
    point_var: Mapping[str, A2[F]],
    cell_var: Mapping[str, A2[F]],
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
    x = convert_3d(x)
    dataarr.add_data(x.astype(np.float64))

    cell = piece.create_elem(XMLElement("Cells"))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii"),
    )
    dataarr.add_data(top.t.astype(np.int64), order=get_vtk_elem(top.vtkelementtype).connectivity)
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii"),
    )
    dataarr.add_data(np.arange(top.nc, top.nc * (top.ne + 1), top.nc, dtype=np.int64))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int8", Name="types", Format="ascii"),
    )
    dataarr.add_data(np.full((top.ne,), top.vtkelementtype.value.idx, dtype=np.int8))
    points = piece.create_elem(XMLElement("PointData", Scalars="scalars"))
    for v, dv in point_var.items():
        dataarr = points.create_elem(
            XMLElement(
                "DataArray",
                type="Float64",
                Name=f"{v}",
                NumberOfComponents=f"{dv.shape[1]}",
                Format="ascii",
            ),
        )
        dataarr.add_data(dv.astype(np.float64))
    if not cell_var:
        return vtkfile
    cells = piece.create_elem(XMLElement("CellData", Scalars="scalars"))
    for v, dv in cell_var.items():
        dataarr = cells.create_elem(
            XMLElement(
                "DataArray",
                type="Float64",
                Name=f"{v}",
                NumberOfComponents=f"{dv.shape[1]}",
                Format="ascii",
            ),
        )
        dataarr.add_data(dv.astype(np.float64))
    return vtkfile


def export_mesh_iter[F: np.floating, I: np.integer](
    args: XMLDataInputs[F, I],
    log: ILogger,
) -> None:
    x, point_var, cell_var = get_xml_variables(args)
    vtk_xml = create_xml_for_mesh(args.prefix, args.top, x, point_var, cell_var)
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
        bart.next() if bart else log.disp(f"<<< Completed {arg.path}")


def run_exports_in_parallel[F: np.floating, I: np.integer](
    mpi: ThreadMethods,
    inp: ProgramArgs,
    indexer: IIndexIterator,
    cache: VariableCache[F, I],
    log: ILogger,
) -> None:
    bart = ProgressBar(len(indexer)) if inp.prog_bar else None
    silent_logger = get_logger("thread", level="NULL")
    with ThreadedRunner(**mpi, prog_bar=bart) as executor:
        for arg in get_arguments(inp, cache, indexer, log=log):
            executor.submit(export_mesh_iter, arg, log=silent_logger)
