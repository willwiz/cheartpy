__all__ = []
import os

import numpy as np
from typing import Sequence
from ..var_types import *
from ..io.indexing import IIndexIterator, get_file_name_indexer
from ..tools.basiclogging import BLogger, ILogger
from .interfaces import *
from .variable_naming import *

# from .third_party import compress_vtu
# from concurrent import futures
# from ..var_types import i32, f64, Arr
# from ..cheart_mesh.io import *
# from ..xmlwriter.xmlclasses import XMLElement, XMLWriters

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


def get_space_data(space: Mat[f64], disp: Mat[f64] | None) -> Arr[tuple[int, int], f64]:
    fx = space if disp is None else space + disp
    # VTU files are defined in 3D space, so we have to append a zero column for 2D data
    if fx.shape[1] == 1:
        raise ValueError(">>>ERROR: Cannot convert data that lives on 1D domains.")
    elif fx.shape[1] == 2:
        z = np.zeros((fx.shape[0], 3), dtype=float)
        z[:, :2] = fx
        return z
    return fx


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


# def create_XML_for_boundary(
#     prefix: str,
#     fx: Arr[tuple[int, int], f64],
#     tp: CheartTopology,
#     fb: Arr[tuple[int, int], i32],
#     fbid: Arr[tuple[int], i32],
# ) -> XMLElement:
#     vtkfile = XMLElement("VTKFile", type="UnstructuredGrid")
#     grid = vtkfile.add_elem(XMLElement("UnstructuredGrid"))
#     piece = grid.add_elem(
#         XMLElement(
#             "Piece",
#             Name=f"{prefix}",
#             NumberOfPoints=f"{len(fx)}",
#             NumberOfCells=f"{fb.shape[0]}",
#         )
#     )
#     dataarr = piece.add_elem(XMLElement("Points")).add_elem(
#         XMLElement("DataArray", type="Float64", NumberOfComponents="3", Format="ascii")
#     )
#     dataarr.add_data(fx, XMLWriters.PointWriter)
#     cell = piece.add_elem(XMLElement("CellData", Scalars="scalars"))
#     dataarr = cell.add_elem(
#         XMLElement("DataArray", type="Int8", Name="PatchIDs", Format="ascii")
#     )
#     dataarr.add_data(fbid, XMLWriters.IntegerWriter)
#     cell = piece.add_elem(XMLElement("Cells", Scalars="scalars"))
#     dataarr = cell.add_elem(
#         XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii")
#     )
#     dataarr.add_data(fb, tp.vtksurfacetype.write)
#     dataarr = cell.add_elem(
#         XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii")
#     )
#     dataarr.add_data(
#         np.arange(fb.shape[1], fb.size + 1, fb.shape[1]), XMLWriters.IntegerWriter
#     )
#     dataarr = cell.add_elem(
#         XMLElement("DataArray", type="Int8", Name="types", Format="ascii")
#     )
#     dataarr.add_data(
#         np.full((fb.shape[0],), tp.vtkelementtype.vtksurfaceid),
#         XMLWriters.IntegerWriter,
#     )
#     return vtkfile
def retrieve_space_data(
    inp: ProgramArgs, time: int | str, cache: VariableCache
) -> Mat[f64]:
    if isinstance(inp.space, CheartMeshFormat):
        fx = cache.nodes
    else:
        fx = inp.space[time]
        if os.path.isfile(fx):
            cache.space = fx
        else:
            fx = cache.space
    # if deformed space, then add displacement
    if inp.disp is None:
        return get_space_data(fx, None)
    fd = inp.disp[time]
    if os.path.isfile(fd):
        cache.disp = fd
    else:
        fd = cache.disp
    return fx, fd


def export_boundary(
    inp: ProgramArgs, cache: VariableCache, LOG: ILogger = BLogger("INFO")
) -> None:
    LOG.debug("<<< Working on", inp.bfile)
    if inp.bfile is None:
        LOG.info(">>> NOTICE: No boundary file given, export is skipped")
        return
    fx, fd = retrieve_space_data(inp, cache.i0, cache)
    dx = get_space_data(fx, fd)
    raw = CHRead_b_utf(inp.bfile)
    db = raw[:, 1:-1]
    dbid = raw[:, -1]
    vtkXML = create_XML_for_boundary(inp.prefix, dx, cache.top, db, dbid)
    foutfile = os.path.join(inp.output_folder, f"{inp.prefix}_boundary.vtu")
    with open(foutfile, "w") as fout:
        vtkXML.write(fout)
    if inp.compression:
        compress_vtu(foutfile, verbose=inp.verbose)
    print("<<< Exported the boundary to", foutfile)


# def import_mesh_data(args: InputArguments, binary: bool = False):
#     fx = get_space_data(args.space, args.disp)
#     variables: dict[str, Arr[tuple[int, int], f64]] = dict()
#     for s, v in args.var.items():
#         # if binary:
#         #     fv = CHRead_d_bin(v)
#         # else:
#         #     fv = CHRead_d_utf(v)
#         fv = CHRead_d(v)
#         if fv.ndim == 1:
#             fv = fv[:, np.newaxis]
#         if fx.shape[0] != fv.shape[0]:
#             raise AssertionError(
#                 f">>>ERROR: Number of values for {
#                                  s} does not match Space."
#             )
#         # append zero column if need be
#         if fv.shape[1] == 2:
#             z = np.zeros((fv.shape[0], 3), dtype=float)
#             z[:, :2] = fv
#             variables[s] = z
#         else:
#             variables[s] = fv
#     return fx, variables


# def create_XML_for_mesh(
#     prefix: str,
#     tp: CheartTopology,
#     fx: Arr[tuple[int, int], f64],
#     var: dict[str, Arr[tuple[int, int], f64]],
# ) -> XMLElement:
#     vtkfile = XMLElement("VTKFile", type="UnstructuredGrid")
#     grid = vtkfile.add_elem(XMLElement("UnstructuredGrid"))
#     piece = grid.add_elem(
#         XMLElement(
#             "Piece",
#             Name=f"{prefix}",
#             NumberOfPoints=f"{fx.shape[0]}",
#             NumberOfCells=f"{tp.ne}",
#         )
#     )
#     points = piece.add_elem(XMLElement("Points"))
#     dataarr = points.add_elem(
#         XMLElement("DataArray", type="Float64", NumberOfComponents="3", Format="ascii")
#     )
#     dataarr.add_data(fx, XMLWriters.PointWriter)

#     cell = piece.add_elem(XMLElement("Cells"))
#     dataarr = cell.add_elem(
#         XMLElement("DataArray", type="Int64", Name="connectivity", Format="ascii")
#     )
#     dataarr.add_data(tp.get_data(), tp.vtkelementtype.write)
#     dataarr = cell.add_elem(
#         XMLElement("DataArray", type="Int64", Name="offsets", Format="ascii")
#     )
#     dataarr.add_data(
#         np.arange(tp.nc, tp.nc * (tp.ne + 1), tp.nc), XMLWriters.IntegerWriter
#     )
#     dataarr = cell.add_elem(
#         XMLElement("DataArray", type="Int8", Name="types", Format="ascii")
#     )
#     dataarr.add_data(
#         np.full((tp.ne,), tp.vtkelementtype.vtkelementid), XMLWriters.IntegerWriter
#     )

#     points = piece.add_elem(XMLElement("PointData", Scalars="scalars"))

#     for v, dv in var.items():
#         dataarr = points.add_elem(
#             XMLElement(
#                 "DataArray",
#                 type="Float64",
#                 Name=f"{v}",
#                 NumberOfComponents=f"{dv.shape[1]}",
#                 Format="ascii",
#             )
#         )
#         dataarr.add_data(dv, XMLWriters.FloatArrWriter)
#     return vtkfile


# def export_mesh_iter(
#     args: InputArguments,
#     inp: ProgramArgs,
#     tp: CheartTopology,
# ) -> None:
#     if inp.verbose:
#         print("<<< Working on", args.prefix)
#     fx, var = import_mesh_data(args, inp.binary)
#     vtkXML = create_XML_for_mesh(inp.prefix, tp, fx, var)
#     with open(args.prefix, "w") as fout:
#         vtkXML.write(fout)
#     if inp.compression:
#         compress_vtu(args.prefix, verbose=inp.verbose)


# def find_args_iter(inp: ProgramArgs, time: str, cache: VariableCache):
#     var: dict[str, str] = dict()
#     for v, fn in inp.var.items():
#         name = fn.get_name(time)
#         if os.path.exists(name):
#             cache.var[v] = name
#         else:
#             name = cache.var[v]
#         var[v] = name
#     space, disp = find_space_filenames(inp, time, cache)
#     return InputArguments(
#         space, disp, var, os.path.join(inp.output_folder, f"{inp.prefix}-{time}.vtu")
#     )


# def run_exports_in_series(
#     inp: ProgramArgs, indexer: IndexerList, cache: VariableCache
# ) -> None:
#     time_steps = indexer.get_generator()
#     bart = ProgressBar(indexer.size, "Exporting") if inp.progress_bar else None
#     for t in time_steps:
#         args = find_args_iter(inp, t, cache)
#         export_mesh_iter(args, inp, cache.top)
#         if bart:
#             bart.next()
#         else:
#             print(f"<<< Completed {args.prefix}")


# def run_exports_in_parallel(
#     inp: ProgramArgs, indexer: IndexerList, cache: VariableCache
# ) -> None:
#     time_steps = indexer.get_generator()
#     args: PARALLELEXEC_ARGS = [
#         ([find_args_iter(inp, t, cache), inp, cache.top], dict()) for t in time_steps
#     ]
#     with futures.ProcessPoolExecutor(inp.cores) as exec:
#         parallel_exec(exec, export_mesh_iter, args, True)
