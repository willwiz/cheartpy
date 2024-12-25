from ..cheart_mesh.io import fix_suffix
from ..tools.basiclogging import BLogger, ILogger
from .interfaces import CmdLineArgs
from ..io.indexing import SearchMode, IIndexIterator


def print_guard(LOG: ILogger = BLogger("INFO")) -> None:
    LOG.disp(
        "\n################################################################################################\n"
    )


def print_header(LOG: ILogger = BLogger("INFO")) -> None:
    LOG.disp(
        "################################################################################################"
    )
    LOG.disp("    Program for converting CHeart data to vtk unstructured grid format")
    LOG.disp("    author: Andreas Hessenthaler")
    LOG.disp("    modified by: Will Zhang")
    LOG.disp("    data: 12/24/2024")
    LOG.disp(
        "################################################################################################\n"
    )


def print_input_info(inp: CmdLineArgs, LOG: ILogger = BLogger("INFO")) -> None:
    LOG.disp(f"The retrieving data from ", inp.input_dir)
    match inp.mesh:
        case str():
            LOG.disp(f"<<< Running Program with Mode: find")
            LOG.disp(f"The prefix for the mesh to use is", fix_suffix(inp.mesh))
        case (x, t, b):
            LOG.disp(f"<<< Running Program with Mode: index")
            LOG.disp(f"The space file to use is ", x)
            LOG.disp(f"The topology file to use is ", t)
            LOG.disp(f"The boundary file to use is ", b)
    LOG.disp(f"<<< The varibles to add are: ")
    LOG.disp(inp.var)
    match inp.index:
        case SearchMode.none:
            LOG.disp(f"No variable will be used for this run")
        case SearchMode.auto:
            LOG.disp(f"<<< Attempting to find time steps from variable file names")
        case (i, j, k):
            LOG.disp(f"<<< Time step: From {i} to {j} in steps of {k}")
    match inp.subindex:
        case SearchMode.none:
            pass
        case SearchMode.auto:
            LOG.disp(f"<<< Automatically finding subiterations")
        case (i, j, k):
            LOG.disp(f"<<< Sub iterations: From {i} to {j} in steps of {k}")
    LOG.disp(f"<<< Output file name prefix: {inp.prefix}")
    LOG.disp(f"<<< Output folder:           {inp.output_dir}")
    LOG.disp(f"<<< Compress VTU:            {inp.compression}")
    LOG.disp(f"<<< Import data as binary:   {inp.binary}")
    if inp.time_series is not None:
        LOG.disp(f"<<< Adding time series from {inp.time_series}")


def print_index_info(indexer: IIndexIterator, LOG: ILogger = BLogger("INFO")) -> None:
    first = last = next(iter(indexer))
    for last in indexer:
        pass
    LOG.disp(f"<<<     Time step found: From {first} to {last}")
