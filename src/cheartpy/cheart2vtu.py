from cheartpy.cheart2vtu_core.core import (
    export_boundary,
    init_variable_cache,
    parse_cmdline_args,
    run_exports_in_series,
    run_exports_in_parallel,
)
from cheartpy.cheart2vtu_core.print_headers import (
    print_guard,
    print_header,
    print_index_info,
)


def main_cli(cmd_args: list[str] | None = None) -> None:
    print_header()
    inp, indexer = parse_cmdline_args(cmd_args)
    print_index_info(indexer)
    print_guard()
    cache = init_variable_cache(inp, indexer)
    export_boundary(inp, cache)
    if inp.cores > 1:
        run_exports_in_parallel(inp, indexer, cache)
    else:
        run_exports_in_series(inp, indexer, cache)
    print_guard()


if __name__ == "__main__":
    main_cli()
