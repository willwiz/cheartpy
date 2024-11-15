from typing import Mapping
from ..trait import *


def recurse_get_var_list_var(var: IVariable) -> Mapping[str, IVariable]:
    expr_deps = [recurse_get_var_list_expr(e) for e in var.get_expr_deps()]
    _all_var_ = {k: v for d in expr_deps for k, v in d.items()}
    if str(var) not in _all_var_:
        _all_var_[str(var)] = var
    return _all_var_


def recurse_get_var_list_expr(expr: IExpression) -> Mapping[str, IVariable]:
    var_deps = [recurse_get_var_list_var(v) for v in expr.get_var_deps()]
    expr_deps = [recurse_get_var_list_expr(e) for e in expr.get_expr_deps()]
    _all_var_ = {k: v for d in expr_deps + var_deps for k, v in d.items()}
    return _all_var_
