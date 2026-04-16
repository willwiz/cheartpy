from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

    from cheartpy.fe.trait import (
        IBCPatch,
        IBoundaryCondition,
        IExpression,
        ILaw,
        IProblem,
        IVariable,
    )


def bcpatch_get_var_deps(patch: IBCPatch) -> Collection[IVariable]: ...
def bcpatch_get_expr_deps(patch: IBCPatch) -> Collection[IExpression]: ...


def bc_get_var_deps(bc: IBoundaryCondition) -> Collection[IVariable]: ...
def bc_get_expr_deps(bc: IBoundaryCondition) -> Collection[IExpression]: ...


def prob_get_var_deps(prob: IProblem) -> Collection[IVariable]: ...
def prob_get_expr_deps(prob: IProblem) -> Collection[IExpression]: ...


def matlaw_get_var_deps(matlaw: ILaw) -> Collection[IVariable]: ...
def matlaw_get_expr_deps(matlaw: ILaw) -> Collection[IExpression]: ...
