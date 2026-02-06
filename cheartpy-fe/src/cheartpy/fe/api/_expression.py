from typing import TYPE_CHECKING

from cheartpy.fe.impl import Expression

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cheartpy.fe.trait import EXPRESSION_VALUE, IExpression


def create_expr(name: str, value: Sequence[EXPRESSION_VALUE]) -> IExpression:
    return Expression(name, value)
