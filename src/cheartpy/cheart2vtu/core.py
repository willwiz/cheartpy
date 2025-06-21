__all__ = [
    "export_boundary",
    "run_exports_in_parallel",
    "run_exports_in_series",
]
import os
from collections.abc import Mapping
from concurrent import futures

import numpy as np
from arraystubs import Arr1, Arr2

s
from ..cheart_mesh.io import chread_b_utf
from ..io.indexing import IIndexIterator
from ..tools.basiclogging import BLogger, ILogger
from ..tools.parallel_exec import PEXEC_ARGS, parallel_exec
from ..tools.progress_bar import ProgressBar
from ..xmlwriter import XMLElement, XMLWriters
from .fio import update_variable_cache
from .third_party import compress_vtu
from .trait import *
from .variable_naming import CheartVTUFormat


def create_XML_for_boundary[I: np.integer, F: np.floating](
    prefix: str,
    fx: Arr2[F],
    tp: CheartTopology,
    fb: Arr2[I],
    fbid: Arr1[I],
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
    dataarr.add_data(fx, XMLWriters.array_writer)
    cell = piece.create_elem(XMLElement("CellData", Scalars="scalars"))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int8", Name="PatchIDs", Format="ascii"),
    )
    dataarr.add_data(fbid, XMLWriters.int_writer)
    cell = piece.create_elem(XMLElement("Cells", Scalars="scalars"))
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii"),
    )
    dataarr.add_data(fb, tp.vtksurfacetype.write)
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii"),
    )
    dataarr.add_data(
        np.arange(fb.shape[1], fb.size + 1, fb.shape[1]),
        XMLWriters.int_writer,
    )
    dataarr = cell.create_elem(
        XMLElement("DataArray", type="Int8", Name="types", Format="ascii"),
    )
    dataarr.add_data(
        np.full((fb.shape[0],), tp.vtkelementtype.vtksurfaceid),
        XMLWriters.int_writer,
    )
    return vtkfile


def export_boundary(
    inp: ProgramArgs,
    cache: VariableCache,
    LOG: ILogger = BLogger("INFO"),
) -> None:
    LOG.debug("<<< Working on", inp.bfile)
    if inp.bfile is None:
        LOG.info(">>> NOTICE: No boundary file given, export is skipped")
        return
    dx = cache.space
    raw = chread_b_utf(inp.bfile)
    db = raw[:, 1:-1]
    dbid = raw[:, -1]
    vtkXML = create_XML_for_boundary(inp.prefix, dx, cache.top, db, dbid)
    foutfile = os.path.join(inp.output_folder, f"{inp.prefix}_boundary.vtu")
    with open(foutfile, "w") as fout:
        vtkXML.write(fout)
    if inp.compression:
        compress_vtu(foutfile, LOG=LOG)
    LOG.disp(f"<<< Exported the boundary to {foutfile}")


def create_XML_for_mesh(
    prefix: str,
    tp: CheartTopology,
    fx: Mat[f64],
    var: Mapping[str, Mat[f64]],
) -> XMLElement:
    vtkfile = XMLElement("VTKFile", type="UnstructuredGrid")
    grid = vtkfile.add_elem(XMLElement("UnstructuredGrid"))
    piece = grid.add_elem(
        XMLElement(
            "Piece",
            Name=f"{prefix}",
            NumberOfPoints=f"{fx.shape[0]}",
            NumberOfCells=f"{tp.ne}",
        ),
    )
    points = piece.add_elem(XMLElement("Points"))
    dataarr = points.add_elem(
        XMLElement("DataArray", type="Float64", NumberOfComponents="3", Format="ascii"),
    )
    dataarr.add_data(fx, XMLWriters.PointWriter)

    cell = piece.add_elem(XMLElement("Cells"))
    dataarr = cell.add_elem(
        XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii"),
    )
    dataarr.add_data(tp.get_data(), tp.vtkelementtype.write)
    dataarr = cell.add_elem(
        XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii"),
    )
    dataarr.add_data(
        np.arange(tp.nc, tp.nc * (tp.ne + 1), tp.nc),
        XMLWriters.IntegerWriter,
    )
    dataarr = cell.add_elem(
        XMLElement("DataArray", type="Int8", Name="types", Format="ascii"),
    )
    dataarr.add_data(
        np.full((tp.ne,), tp.vtkelementtype.vtkelementid),
        XMLWriters.IntegerWriter,
    )
    points = piece.add_elem(XMLElement("PointData", Scalars="scalars"))
    for v, dv in var.items():
        dataarr = points.add_elem(
            XMLElement(
                "DataArray",
                type="Float64",
                Name=f"{v}",
                NumberOfComponents=f"{dv.shape[1]}",
                Format="ascii",
            ),
        )
        dataarr.add_data(dv, XMLWriters.FloatArrWriter)
    return vtkfile


def export_mesh_iter(
    prefix: str,
    t: str | int,
    inp: ProgramArgs,
    cache: VariableCache,
    LOG: ILogger = BLogger("INFO"),
) -> None:
    LOG.debug("<<< Working on", prefix)
    x, vars = update_variable_cache(inp, t, cache, LOG)
    LOG.debug("<<< showing cache")
    LOG.debug(cache)
    vtkXML = create_XML_for_mesh(inp.prefix, cache.top, x, vars)
    with open(prefix, "w") as fout:
        vtkXML.write(fout)
    if inp.compression:
        compress_vtu(prefix, LOG)


def run_exports_in_series(
    inp: ProgramArgs,
    indexer: IIndexIterator,
    cache: VariableCache,
    LOG: ILogger = BLogger("INFO"),
) -> None:
    LOG.disp("Processing vtus")
    name = CheartVTUFormat(inp.output_folder, inp.prefix)
    bart = ProgressBar(len(indexer)) if inp.progress_bar else None
    for t in indexer:
        export_mesh_iter(name[t], t, inp, cache, LOG)
        bart.next() if bart else print(f"<<< Completed {name[t]}")


def run_exports_in_parallel(
    inp: ProgramArgs,
    indexer: IIndexIterator,
    cache: VariableCache,
    LOG: ILogger = BLogger("INFO"),
) -> None:
    LOG.disp("Processing vtus")
    name = CheartVTUFormat(inp.output_folder, inp.prefix)
    args: PEXEC_ARGS = [([name[t], t, inp, cache, LOG], dict()) for t in indexer]
    with futures.ProcessPoolExecutor(inp.cores) as exec:
        parallel_exec(exec, export_mesh_iter, args, True)
