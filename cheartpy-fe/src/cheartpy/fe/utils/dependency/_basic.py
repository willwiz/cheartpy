from typing import TYPE_CHECKING, Unpack

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from collections.abc import Collection

    from cheartpy.fe.trait import IExpression, IVariable


class Exclusions(TypedDict, total=False):
    exclude_var: set[IVariable]
    exclude_expr: set[IExpression]


def variable_get_var_deps(v: IVariable, **kwargs: Unpack[Exclusions]) -> Collection[IVariable]:
    self_exprs = set(v.expr_deps.values())
    var_excludes: set[IVariable] = kwargs.get("exclude_var", set())
    expr_excludes: set[IExpression] = kwargs.get("exclude_expr", set())
    return {
        var
        for e in self_exprs - expr_excludes
        for var in expression_get_var_deps(
            e, exclude_var=var_excludes, exclude_expr=self_exprs | expr_excludes
        )
    }


def variable_get_expr_deps(v: IVariable, **kwargs: Unpack[Exclusions]) -> Collection[IExpression]:
    self_exprs = set(v.expr_deps.values())
    var_excludes: set[IVariable] = kwargs.get("exclude_var", set())
    expr_excludes: set[IExpression] = kwargs.get("exclude_expr", set())
    from_expr_deps = {
        expr
        for e in self_exprs - expr_excludes
        for expr in expression_get_expr_deps(
            e, exclude_var=var_excludes, exclude_expr=self_exprs | expr_excludes
        )
    }
    return self_exprs | from_expr_deps


def expression_get_var_deps(e: IExpression, **kwargs: Unpack[Exclusions]) -> Collection[IVariable]:
    self_var = set(e.var_deps.values())
    self_expr = set(e.expr_deps.values())
    var_excludes: set[IVariable] = kwargs.get("exclude_var", set())
    expr_excludes: set[IExpression] = kwargs.get("exclude_expr", set())
    updated_var_excludes = self_var | var_excludes
    updated_expr_excludes = self_expr | expr_excludes
    from_var_deps = {
        var
        for v in self_var - var_excludes
        for var in variable_get_var_deps(
            v, exclude_var=updated_var_excludes, exclude_expr=updated_expr_excludes
        )
    }
    from_expr_deps = {
        var
        for e in self_expr - expr_excludes
        for var in expression_get_var_deps(
            e, exclude_var=updated_var_excludes, exclude_expr=updated_expr_excludes
        )
    }
    return self_var | from_var_deps | from_expr_deps


def expression_get_expr_deps(
    e: IExpression, **kwargs: Unpack[Exclusions]
) -> Collection[IExpression]:
    self_var = set(e.var_deps.values())
    self_expr = set(e.expr_deps.values())
    var_excludes: set[IVariable] = kwargs.get("exclude_var", set())
    expr_excludes: set[IExpression] = kwargs.get("exclude_expr", set())
    updated_var_excludes = self_var | var_excludes
    updated_expr_excludes = self_expr | expr_excludes
    from_var_deps = {
        expr
        for v in self_var - var_excludes
        for expr in variable_get_expr_deps(
            v, exclude_var=updated_var_excludes, exclude_expr=updated_expr_excludes
        )
    }
    from_expr_deps = {
        expr
        for e in self_expr - expr_excludes
        for expr in expression_get_expr_deps(
            e, exclude_var=updated_var_excludes, exclude_expr=updated_expr_excludes
        )
    }
    return self_expr | from_var_deps | from_expr_deps
