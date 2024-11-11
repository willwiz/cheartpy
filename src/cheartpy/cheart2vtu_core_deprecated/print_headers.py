from .data_types import CmdLineArgs
from .file_indexing import DFileAutoFinder, IndexerList


def print_guard() -> None:
    print(
        "\n################################################################################################\n"
    )


def print_header() -> None:
    print(
        "################################################################################################"
    )
    print("    Program for converting CHeart data to vtk unstructured grid format")
    print("    author: Andreas Hessenthaler")
    print("    modified by: Will Zhang")
    print("    data: 10/17/2023")
    print(
        "################################################################################################\n"
    )


def print_input_info(inp: CmdLineArgs) -> None:
    print(f"<<< Running Program with Mode: {inp.cmd}")
    print(f"The retrieving data from ", inp.input_folder)
    print(f"The space file to use is ", inp.xfile)
    print(f"The topology file to use is ", inp.tfile)
    print(f"The boundary file to use is ", inp.bfile)
    if inp.bfile is not None:
        print(f"<<< Output file name (boundary): {inp.prefix}_boundary.vtu")
    print(f"The varibles to add are: ")
    print(inp.var)
    if inp.index is not None:
        print(
            f"<<< Time step: From {inp.index[0]} to {
                inp.index[1]} with steps of {inp.index[2]}"
        )
    else:
        print(f"<<< Attempting to find time steps from variable file names")
    if inp.sub_index is not None:
        print(
            f"<<< Sub iterations: From {inp.sub_index[0]} to {
                inp.sub_index[1]} with steps of {inp.sub_index[2]}"
        )
    if inp.sub_auto:
        print(f"<<< Automatically finding subiterations")
    print(f"<<< Output file name prefix: {inp.prefix}")
    print(f"<<< Output folder:           {inp.output_folder}")
    print(f"<<< Compress VTU:            {inp.compression}")
    print(f"<<< Import data as binary:   {inp.binary}")
    if inp.time_series is not None:
        print(f"<<< Adding time series from {inp.time_series}")


def print_index_info(indexer: IndexerList) -> None:
    if isinstance(indexer, DFileAutoFinder):
        gen = indexer.get_generator()
        first = last = next(gen)
        for last in gen:
            pass
        print(f"<<<     Time step found: From {first} to {last}")
