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
    LOG.disp("    data: 10/17/2023")
    LOG.disp(
        "################################################################################################\n"
    )


def print_input_info(inp: CmdLineArgs, LOG: ILogger = BLogger("INFO")) -> None:
    LOG.disp(f"<<< Running Program with Mode: {inp.cmd}")
    LOG.disp(f"The retrieving data from ", inp.input_folder)
    LOG.disp(f"The space file to use is ", inp.xfile)
    LOG.disp(f"The topology file to use is ", inp.tfile)
    LOG.disp(f"The boundary file to use is ", inp.bfile)
    if inp.bfile is not None:
        LOG.disp(f"<<< Output file name (boundary): {inp.prefix}_boundary.vtu")
    LOG.disp(f"The varibles to add are: ")
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
    LOG.disp(f"<<< Output folder:           {inp.output_folder}")
    LOG.disp(f"<<< Compress VTU:            {inp.compression}")
    LOG.disp(f"<<< Import data as binary:   {inp.binary}")
    if inp.time_series is not None:
        LOG.disp(f"<<< Adding time series from {inp.time_series}")


def print_index_info(indexer: IIndexIterator, LOG: ILogger = BLogger("INFO")) -> None:
    first = last = next(iter(indexer))
    for last in indexer:
        pass
    LOG.disp(f"<<<     Time step found: From {first} to {last}")
