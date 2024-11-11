__all__ = [
    "update_variable_cache",
    "export_boundary",
    "run_exports_in_series",
    "run_exports_in_parallel",
]
from concurrent import futures
import os

from ..tools.parallel_exec import PEXEC_ARGS, parallel_exec
from ..tools.progress_bar import ProgressBar
import numpy as np
from typing import TypeIs
from ..var_types import *
from ..io.indexing import IIndexIterator
from ..tools.basiclogging import BLogger, ILogger
from .interfaces import *
from .variable_naming import *

from .third_party import compress_vtu

# from concurrent import futures
# from ..var_types import i32, f64, Arr
from ..cheart_mesh.io import *

from ..xmlwriter.xmlclasses import XMLElement, XMLWriters

# # from ..cheart2vtu_core.print_headers import print_input_info
# # from ..cheart2vtu_core.main_parser import get_cmdline_args
# # from ..cheart2vtu_core.file_indexing import (
# #     IndexerList,
# #     get_file_name_indexer,
# # )
# # from ..cheart2vtu_core.data_types import (
# #     CheartMeshFormat,
# #     CheartVarFormat,
# #     CheartZipFormat,
# #     InputArguments,
# #     ProgramArgs,
# #     VariableCache,
# #     CheartTopology,
# # )
# from ..tools.progress_bar import ProgressBar
# from ..tools.parallel_exec import *
from .print_headers import *
from .variable_naming import CheartVTUFormat


# def find_space_filenames(
#     inp: ProgramArgs, time: int | str, cache: VariableCache
# ) -> tuple[Arr[tuple[int, int], f64] | str, None | str]:
#     if isinstance(inp.space, CheartMeshFormat):
#         fx = cache.nodes
#     else:
#         fx = inp.space.get_name(time)
#         if os.path.isfile(fx):
#             cache.space = fx
#         else:
#             fx = cache.space
#     # if deformed space, then add displacement
#     if inp.disp is None:
#         return fx, None
#     fd = inp.disp.get_name(time)
#     if os.path.isfile(fd):
#         cache.disp = fd
#     else:
#         fd = cache.disp
#     return fx, fd


def create_XML_for_boundary(
    prefix: str,
    fx: Arr[tuple[int, int], f64],
    tp: CheartTopology,
    fb: Arr[tuple[int, int], i32],
    fbid: Arr[tuple[int], i32],
) -> XMLElement:
    vtkfile = XMLElement("VTKFile", type="UnstructuredGrid")
    grid = vtkfile.add_elem(XMLElement("UnstructuredGrid"))
    piece = grid.add_elem(
        XMLElement(
            "Piece",
            Name=f"{prefix}",
            NumberOfPoints=f"{len(fx)}",
            NumberOfCells=f"{fb.shape[0]}",
        )
    )
    dataarr = piece.add_elem(XMLElement("Points")).add_elem(
        XMLElement("DataArray", type="Float64", NumberOfComponents="3", Format="ascii")
    )
    dataarr.add_data(fx, XMLWriters.PointWriter)
    cell = piece.add_elem(XMLElement("CellData", Scalars="scalars"))
    dataarr = cell.add_elem(
        XMLElement("DataArray", type="Int8", Name="PatchIDs", Format="ascii")
    )
    dataarr.add_data(fbid, XMLWriters.IntegerWriter)
    cell = piece.add_elem(XMLElement("Cells", Scalars="scalars"))
    dataarr = cell.add_elem(
        XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii")
    )
    dataarr.add_data(fb, tp.vtksurfacetype.write)
    dataarr = cell.add_elem(
        XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii")
    )
    dataarr.add_data(
        np.arange(fb.shape[1], fb.size + 1, fb.shape[1]), XMLWriters.IntegerWriter
    )
    dataarr = cell.add_elem(
        XMLElement("DataArray", type="Int8", Name="types", Format="ascii")
    )
    dataarr.add_data(
        np.full((fb.shape[0],), tp.vtkelementtype.vtksurfaceid),
        XMLWriters.IntegerWriter,
    )
    return vtkfile


def not_none[T](var: T | None) -> TypeIs[T]:
    return var is not None


def update_variable_cache(
    inp: ProgramArgs,
    time: int | str,
    cache: VariableCache,
    LOG: ILogger = BLogger("NULL"),
):
    if time == cache.t:
        LOG.debug(f"time point {time} did not change")
        return cache
    fx = inp.space[time]
    update_space = fx != cache.space_i
    if update_space:
        LOG.debug(f"updating space to file {fx}")
        cache.space = CHRead_d(fx)
        cache.space_i = fx
    if inp.disp is None:
        update_disp = False
    else:
        fd = inp.disp[time]
        update_disp = fd != cache.disp_i
        if update_disp:
            LOG.debug(f"updating disp to file {fd}")
            cache.disp = CHRead_d(fd)
            cache.disp_i = fd
    match update_space, update_disp:
        case False, False:
            pass
        case True, False:
            cache.x = cache.space
        case _, True:
            cache.x = cache.space + cache.disp
    for k, var in inp.var.items():
        new_v = var[time]
        LOG.debug(f"updating var {k} to file {new_v} from {cache.var_i[k]}")
        if (cache.var_i[k] != new_v) and os.path.isfile(new_v):
            LOG.debug(f"updating var {k} to file {new_v}")
            cache.var[k] = CHRead_d(new_v)
            cache.var_i[k] = new_v
    return cache


def export_boundary(
    inp: ProgramArgs, cache: VariableCache, LOG: ILogger = BLogger("INFO")
) -> None:
    LOG.debug("<<< Working on", inp.bfile)
    if inp.bfile is None:
        LOG.info(">>> NOTICE: No boundary file given, export is skipped")
        return
    dx = cache.disp
    raw = CHRead_b_utf(inp.bfile)
    db = raw[:, 1:-1]
    dbid = raw[:, -1]
    vtkXML = create_XML_for_boundary(inp.prefix, dx, cache.top, db, dbid)
    foutfile = os.path.join(inp.output_folder, f"{inp.prefix}_boundary.vtu")
    with open(foutfile, "w") as fout:
        vtkXML.write(fout)
    if inp.compression:
        compress_vtu(foutfile, LOG=LOG)
    LOG.info("<<< Exported the boundary to", foutfile)


def create_XML_for_mesh(
    prefix: str,
    tp: CheartTopology,
    fx: Arr[tuple[int, int], f64],
    var: dict[str, Mat[f64]],
) -> XMLElement:
    vtkfile = XMLElement("VTKFile", type="UnstructuredGrid")
    grid = vtkfile.add_elem(XMLElement("UnstructuredGrid"))
    piece = grid.add_elem(
        XMLElement(
            "Piece",
            Name=f"{prefix}",
            NumberOfPoints=f"{fx.shape[0]}",
            NumberOfCells=f"{tp.ne}",
        )
    )
    points = piece.add_elem(XMLElement("Points"))
    dataarr = points.add_elem(
        XMLElement("DataArray", type="Float64", NumberOfComponents="3", Format="ascii")
    )
    dataarr.add_data(fx, XMLWriters.PointWriter)

    cell = piece.add_elem(XMLElement("Cells"))
    dataarr = cell.add_elem(
        XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii")
    )
    dataarr.add_data(tp.get_data(), tp.vtkelementtype.write)
    dataarr = cell.add_elem(
        XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii")
    )
    dataarr.add_data(
        np.arange(tp.nc, tp.nc * (tp.ne + 1), tp.nc), XMLWriters.IntegerWriter
    )
    dataarr = cell.add_elem(
        XMLElement("DataArray", type="Int8", Name="types", Format="ascii")
    )
    dataarr.add_data(
        np.full((tp.ne,), tp.vtkelementtype.vtkelementid), XMLWriters.IntegerWriter
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
            )
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
    cache = update_variable_cache(inp, t, cache, LOG)
    LOG.debug("<<< showing cache")
    LOG.debug(cache)
    vtkXML = create_XML_for_mesh(inp.prefix, cache.top, cache.x, cache.var)
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
