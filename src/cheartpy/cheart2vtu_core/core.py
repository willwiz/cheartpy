import os
import meshio
import numpy as np
from concurrent import futures
from ..var_types import i32, f64, Arr
from ..meshing.cheart.io import (
    CHRead_d,
    CHRead_d_utf,
    CHRead_d_bin,
    CHRead_b_utf,
)
from cheartpy.xmlwriter.xmlclasses import XMLElement, XMLWriters
from cheartpy.cheart2vtu_core.print_headers import print_input_info
from cheartpy.cheart2vtu_core.main_parser import get_cmdline_args
from cheartpy.cheart2vtu_core.file_indexing import (
    IndexerList,
    get_file_name_indexer,
)
from cheartpy.cheart2vtu_core.data_types import (
    CheartMeshFormat,
    CheartVarFormat,
    CheartZipFormat,
    InputArguments,
    ProgramArgs,
    VariableCache,
    CheartTopology,
)
from cheartpy.tools.progress_bar import ProgressBar


def compress_vtu(name: str, verbose: bool = False) -> None:
    if verbose:
        size = os.stat(name).st_size
        print("File size before: {:.2f} MB".format(size / 1024**2))
    mesh = meshio._helpers.read(name, file_format="vtu")
    meshio.vtu.write(name, mesh, binary=True, compression="zlib")
    if verbose:
        size = os.stat(name).st_size
        print("File size after: {:.2f} MB".format(size / 1024**2))


def parse_cmdline_args(
    cmd_args: list[str | float | int] | None = None,
) -> tuple[ProgramArgs, IndexerList]:
    err: bool = False
    args = get_cmdline_args([str(v) for v in cmd_args] if cmd_args else None)
    print_input_info(args)
    if not args.input_folder:
        pass
    elif not os.path.isdir(args.input_folder):
        print(f">>>ERROR: Input folder = {args.input_folder} does not exist")
        err = True
    if (args.output_folder != "") and (not os.path.isdir(args.output_folder)):
        os.makedirs(args.output_folder, exist_ok=True)
    mode, indexer = get_file_name_indexer(args)
    generator = indexer.get_generator()
    i0 = next(generator)
    if not os.path.isfile(args.tfile):
        print(
            f">>>ERROR: Topology = {
              args.tfile} cannot be found. Check if correct relative path is given"
        )
        err = True
    if os.path.isfile(args.xfile):
        space = CheartMeshFormat(args.xfile)
    elif os.path.isfile(f"{args.xfile}-{i0}.D"):
        space = CheartVarFormat(args.input_folder, args.xfile)
    elif os.path.isfile(f"{args.xfile}-{i0}.D.gz"):
        space = CheartZipFormat(args.input_folder, args.xfile)
    else:
        print(
            f">>>ERROR: Space = {
              args.xfile} not recognized as mesh, var, or zip"
        )
        err = True

    disp = None
    if args.disp is None:
        pass
    elif os.path.isfile(args.disp):
        disp = CheartMeshFormat(args.disp)
    elif os.path.isfile(f"{args.disp}-{i0}.D"):
        disp = CheartVarFormat(args.input_folder, args.disp)
    elif os.path.isfile(f"{args.disp}-{i0}.D.gz"):
        disp = CheartZipFormat(args.input_folder, args.disp)
    else:
        print(
            f">>>ERROR: Disp = {
              args.disp} not recognized as mesh, var, or zip"
        )
        err = True

    var: dict[str, CheartVarFormat | CheartZipFormat] = dict()
    for v in args.var:
        if os.path.isfile(os.path.join(args.input_folder, f"{v}-{i0}.D")):
            var[v] = CheartVarFormat(args.input_folder, v)
        elif os.path.isfile(os.path.join(args.input_folder, f"{v}-{i0}.D.gz")):
            var[v] = CheartZipFormat(args.input_folder, v)
        else:
            print(f">>>ERROR: Type of {v} cannot be identified.")
            err = True
    if err:
        raise ValueError("At least one error was triggered.")
    return (
        ProgramArgs(
            mode,
            args.prefix,
            args.input_folder,
            args.output_folder,
            args.time_series,
            args.progress_bar if args.var else False,
            args.verbose,
            args.binary,
            args.compression,
            args.cores,
            args.tfile,
            args.bfile,
            space,  # type: ignore : TypeChecker Issue, delay exception is raised if space is None
            disp,
            var,
        ),
        indexer,
    )


def get_space_data(
    space_file: str | Arr[tuple[int, int], f64], disp_file: str | None
) -> Arr[tuple[int, int], f64]:
    if isinstance(space_file, str):
        fx = CHRead_d_utf(space_file)
    else:
        fx = space_file
    if disp_file is not None:
        fx = fx + CHRead_d_utf(disp_file)
    # VTU files are defined in 3D space, so we have to append a zero column for 2D data
    if fx.shape[1] == 1:
        raise ValueError(">>>ERROR: Cannot convert data that lives on 1D domains.")
    elif fx.shape[1] == 2:
        z = np.zeros((fx.shape[0], 3), dtype=float)
        z[:, :2] = fx
        return z
    return fx


def init_variable_cache(inp: ProgramArgs, indexer: IndexerList) -> VariableCache:
    i0 = next(indexer.get_generator())
    top = CheartTopology(inp.tfile, inp.bfile)
    fx = inp.space.get_name(i0)
    fd = None if inp.disp is None else inp.disp.get_name(i0)
    nodes = get_space_data(fx, fd)
    var: dict[str, str] = dict.fromkeys(inp.var.keys(), "")
    for v, fn in inp.var.items():
        name = fn.get_name(i0)
        if os.path.isfile(name):
            var[v] = name
        else:
            raise ValueError(f"The initial value for {v} cannot be found")
    return VariableCache(i0, top, nodes, fx, fd, var)


def find_space_filenames(
    inp: ProgramArgs, time: int | str, cache: VariableCache
) -> tuple[Arr[tuple[int, int], f64] | str, None | str]:
    if isinstance(inp.space, CheartMeshFormat):
        fx = cache.nodes
    else:
        fx = inp.space.get_name(time)
        if os.path.isfile(fx):
            cache.space = fx
        else:
            fx = cache.space
    # if deformed space, then add displacement
    if inp.disp is None:
        return fx, None
    fd = inp.disp.get_name(time)
    if os.path.isfile(fd):
        cache.disp = fd
    else:
        fd = cache.disp
    return fx, fd


def create_XML_for_boundary(
    prefix: str,
    fx: Arr[tuple[int, int], f64],
    tp: CheartTopology,
    fb: Arr[tuple[int, int], i32],
    fbid: Arr[int, i32],
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


def export_boundary(inp: ProgramArgs, cache: VariableCache) -> None:
    if inp.verbose:
        print("<<< Working on", inp.bfile)
    if inp.bfile is None:
        print(">>> NOTICE: No boundary file given, export is skipped")
        return
    fx, fd = find_space_filenames(inp, cache.i0, cache)
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


def import_mesh_data(args: InputArguments, binary: bool = False):
    fx = get_space_data(args.space, args.disp)
    variables: dict[str, Arr[tuple[int, int], f64]] = dict()
    for s, v in args.var.items():
        # if binary:
        #     fv = CHRead_d_bin(v)
        # else:
        #     fv = CHRead_d_utf(v)
        fv = CHRead_d(v)
        if fv.ndim == 1:
            fv = fv[:, np.newaxis]
        if fx.shape[0] != fv.shape[0]:
            raise AssertionError(
                f">>>ERROR: Number of values for {
                                 s} does not match Space."
            )
        # append zero column if need be
        if fv.shape[1] == 2:
            z = np.zeros((fv.shape[0], 3), dtype=float)
            z[:, :2] = fv
            variables[s] = z
        else:
            variables[s] = fv
    return fx, variables


def create_XML_for_mesh(
    prefix: str,
    tp: CheartTopology,
    fx: Arr[tuple[int, int], f64],
    var: dict[str, Arr[tuple[int, int], f64]],
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
    dataarr.add_data(tp._ft, tp.vtkelementtype.write)
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
    args: InputArguments,
    inp: ProgramArgs,
    tp: CheartTopology,
) -> None:
    if inp.verbose:
        print("<<< Working on", args.prefix)
    fx, var = import_mesh_data(args, inp.binary)
    vtkXML = create_XML_for_mesh(inp.prefix, tp, fx, var)
    with open(args.prefix, "w") as fout:
        vtkXML.write(fout)
    if inp.compression:
        compress_vtu(args.prefix, verbose=inp.verbose)


def find_args_iter(inp: ProgramArgs, time: str, cache: VariableCache):
    var: dict[str, str] = dict()
    for v, fn in inp.var.items():
        name = fn.get_name(time)
        if os.path.exists(name):
            cache.var[v] = name
        else:
            name = cache.var[v]
        var[v] = name
    space, disp = find_space_filenames(inp, time, cache)
    return InputArguments(
        space, disp, var, os.path.join(inp.output_folder, f"{inp.prefix}-{time}.vtu")
    )


def run_exports_in_series(
    inp: ProgramArgs, indexer: IndexerList, cache: VariableCache
) -> None:
    time_steps = indexer.get_generator()
    bart = ProgressBar(indexer.size, "Exporting") if inp.progress_bar else None
    for t in time_steps:
        args = find_args_iter(inp, t, cache)
        export_mesh_iter(args, inp, cache.top)
        if bart:
            bart.next()
        else:
            print(f"<<< Completed {args.prefix}")


def run_exports_in_parallel(
    inp: ProgramArgs, indexer: IndexerList, cache: VariableCache
) -> None:
    time_steps = indexer.get_generator()
    jobs: dict[futures.Future, str] = dict()
    bart = ProgressBar(indexer.size, "Exporting") if inp.progress_bar else None
    with futures.ProcessPoolExecutor(inp.cores) as exec:
        for t in time_steps:
            args = find_args_iter(inp, t, cache)
            jobs[exec.submit(export_mesh_iter, args, inp, cache.top)] = args.prefix
        for future in futures.as_completed(jobs):
            try:
                future.result()
                if bart:
                    bart.next()
                else:
                    print(f"<<< Completed {jobs[future]}")
            except Exception as e:
                raise e
