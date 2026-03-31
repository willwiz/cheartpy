from typing import TYPE_CHECKING, overload

from pytools.result import Err, Ok

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


_MASK_ARG_LEN = 3


@overload
def gather_masks(cmd_arg_masks: None) -> Ok[None]: ...
@overload
def gather_masks(
    cmd_arg_masks: Sequence[Sequence[str]],
) -> Ok[Mapping[str, tuple[str, Sequence[str]]]] | Err: ...
def gather_masks(
    cmd_arg_masks: Sequence[Sequence[str]] | None,
) -> Ok[Mapping[str, tuple[str, Sequence[str]]] | None] | Err:
    if cmd_arg_masks is None:
        return Ok(None)
    masks: Mapping[str, tuple[str, Sequence[str]]] = {}
    for m in cmd_arg_masks:
        if len(m) < _MASK_ARG_LEN:
            msg = (
                ">>>ERROR: A Mask must have at least 3 args, e.g., name1, name2, ..., value, output"
            )
            return Err(ValueError(msg))
        masks[m[-1]] = (m[-2], m[:-2])
    return Ok(masks)


@overload
def split_argslist_to_nameddict(varlist: None) -> Ok[None]: ...
@overload
def split_argslist_to_nameddict(
    varlist: Sequence[Sequence[str]],
) -> Ok[dict[int, Sequence[str]]] | Err: ...
def split_argslist_to_nameddict(
    varlist: Sequence[Sequence[str]] | None,
):
    if varlist is None:
        return Ok(None)
    var: dict[int, Sequence[str]] = {}
    for items in varlist:
        if not len(items) > 1:
            msg = ">>>ERROR: Boundary or Topology must have at least 2 items, elem and label."
            return Err(ValueError(msg))
        var[int(items[-1])] = items[:-1]
    return Ok(var)
