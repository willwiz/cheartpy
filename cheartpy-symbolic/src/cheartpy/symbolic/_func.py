from .trait import IExpression, INumber, IOperator, ISymbol


def str_expr(
    *vals: IExpression | ISymbol | INumber | IOperator | float | str | None,
    char: str = " ",
) -> str:
    values = [str(v) for v in vals if v is not None]
    return char.join(values)
