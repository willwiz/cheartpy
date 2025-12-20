# ruff: noqa: PLR0911
from cheartpy.symbolic.trait import (
    ALL_TYPES,
    IMMUTABLE_TYPES,
    ExpressionTrait,
    ExpressionTuple,
    FunctionTrait,
    MathOperator,
    ScaledTrait,
    SymbolTrait,
)


def add_expression(left: ExpressionTrait, right: ALL_TYPES) -> ExpressionTuple:
    match right:
        case float() | int():
            return _add_expression_to_float_(left, right)
        case FunctionTrait() | SymbolTrait():
            return _add_expression_to_functionsymbol_(left, right)
        case ScaledTrait():
            raise NotImplementedError
        case ExpressionTrait():
            return _add_expression_to_expression_(left, right)


def _add_expression_to_float_(left: ExpressionTrait, right: float) -> ExpressionTuple:
    match left.op:
        case MathOperator.ADD | MathOperator.SUB:
            return _add_expression_to_float_addsub_(left, right)
        case MathOperator.MUL:
            return _add_expression_to_float_mul_(left, right)
        case MathOperator.DIV:
            return _add_expression_to_float_div_(left, right)
        case MathOperator.POW:
            return _add_expression_to_float_pow_(left, right)
        case MathOperator.MOD:
            return _add_expression_to_float_mod_(left, right)


def _add_expression_to_float_addsub_(left: ExpressionTrait, right: float) -> ExpressionTuple:
    if left.op != MathOperator.ADD:
        msg = (
            "add_expression_to_float_add_ can only handle addition expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    if isinstance(left.left, float | int):
        return ExpressionTuple(left.left + right, left.op, left.right)
    if isinstance(left.right, float | int):
        return ExpressionTuple(left.left, left.op, left.right + right)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_float_mul_(left: ExpressionTrait, right: float) -> ExpressionTuple:
    if left.op != MathOperator.MUL:
        msg = (
            "add_expression_to_float_mul_ can only handle multiplication expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    if left.left == -1:
        return ExpressionTuple(right, MathOperator.SUB, left.right)
    if left.right == -1:
        return ExpressionTuple(right, MathOperator.SUB, left.left)
    if left.left == 1:
        return ExpressionTuple(left.right, MathOperator.ADD, right)
    if left.right == 1:
        return ExpressionTuple(left.left, MathOperator.ADD, right)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_float_div_(left: ExpressionTrait, right: float) -> ExpressionTuple:
    if left.op != MathOperator.DIV:
        msg = (
            "add_expression_to_float_div_ can only handle division expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_float_pow_(left: ExpressionTrait, right: float) -> ExpressionTuple:
    if left.op != MathOperator.POW:
        msg = (
            "add_expression_to_float_pow_ can only handle power expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    if left.left == right:
        return ExpressionTuple(right, left.op, left.right + 1)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_float_mod_(left: ExpressionTrait, right: float) -> ExpressionTuple:
    if left.op != MathOperator.MOD:
        msg = (
            "add_expression_to_float_mod_ can only handle modulus expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_functionsymbol_(
    left: ExpressionTrait, right: IMMUTABLE_TYPES
) -> ExpressionTuple:
    match left.op:
        case MathOperator.ADD | MathOperator.SUB:
            return _add_expression_to_functionsymbol_addsub_(left, right)
        case MathOperator.MUL:
            return _add_expression_to_functionsymbol_mul_(left, right)
        case MathOperator.DIV:
            return _add_expression_to_functionsymbol_div_(left, right)
        case MathOperator.POW:
            return _add_expression_to_functionsymbol_pow_(left, right)
        case MathOperator.MOD:
            return _add_expression_to_functionsymbol_mod_(left, right)


def _add_expression_to_functionsymbol_addsub_(
    left: ExpressionTrait, right: IMMUTABLE_TYPES
) -> ExpressionTuple:
    if left.op != MathOperator.ADD:
        msg = (
            "add_expression_to_immutable_add_ can only handle addition expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    if left.left == right:
        return ExpressionTuple(left.left + right, left.op, left.right)
    if left.right == right:
        return ExpressionTuple(left.left, left.op, left.right + right)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_functionsymbol_mul_(
    left: ExpressionTrait, right: IMMUTABLE_TYPES
) -> ExpressionTuple:
    if left.op != MathOperator.MUL:
        msg = (
            "add_expression_to_function_mul_ can only handle multiplication expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    if left.left == -1:
        return ExpressionTuple(right, MathOperator.SUB, left.right)
    if left.right == -1:
        return ExpressionTuple(right, MathOperator.SUB, left.left)
    if left.left == 1:
        return ExpressionTuple(left.right, MathOperator.ADD, right)
    if left.right == 1:
        return ExpressionTuple(left.left, MathOperator.ADD, right)
    if left.left == right:
        return ExpressionTuple(right, left.op, left.right + 1)
    if left.right == right:
        return ExpressionTuple(left.left + 1, left.op, right)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_functionsymbol_div_(
    left: ExpressionTrait, right: IMMUTABLE_TYPES
) -> ExpressionTuple:
    if left.op != MathOperator.DIV:
        msg = (
            "add_expression_to_function_div_ can only handle division expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    if left.left == right:
        return ExpressionTuple(right, left.op, 1 / left.right + 1)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_functionsymbol_pow_(
    left: ExpressionTrait, right: IMMUTABLE_TYPES
) -> ExpressionTuple:
    if left.op != MathOperator.POW:
        msg = (
            "add_expression_to_function_pow_ can only handle power expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    if left.left == right:
        return ExpressionTuple(right, left.op, left.right + 1)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_functionsymbol_mod_(
    left: ExpressionTrait, right: IMMUTABLE_TYPES
) -> ExpressionTuple:
    if left.op != MathOperator.MOD:
        msg = (
            "add_expression_to_function_mod_ can only handle modulus expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_expression_(
    left: ExpressionTrait, right: ExpressionTrait
) -> ExpressionTuple:
    match left.op:
        case MathOperator.ADD | MathOperator.SUB:
            return _add_expression_to_expression_add_(left, right)
        case MathOperator.MUL:
            # return _add_expression_to_expression_mul_(left, right)
            raise NotImplementedError
        case MathOperator.DIV:
            # return _add_expression_to_expression_div_(left, right)
            raise NotImplementedError
        case MathOperator.POW:
            # return _add_expression_to_expression_pow_(left, right)
            raise NotImplementedError
        case MathOperator.MOD:
            # return _add_expression_to_expression_mod_(left, right)
            raise NotImplementedError


def _add_add_lsimplifiable(left: ExpressionTrait, right: ExpressionTrait) -> bool:
    # left.op == ADD
    if (isinstance(left.left, float | int) or isinstance(left.right, float | int)) and isinstance(
        right.left, float | int
    ):
        return True
    if left.left == right.left:
        return True
    return left.right == right.left


def _add_add_rsimplifiable(left: ExpressionTrait, right: ExpressionTrait) -> bool:
    # left.op == ADD
    if (isinstance(left.left, float | int) or isinstance(left.right, float | int)) and isinstance(
        right.right, float | int
    ):
        return True
    if left.left == right.right:
        return True
    return left.right == right.right


def _add_expression_to_expression_add_(
    left: ExpressionTrait, right: ExpressionTrait
) -> ExpressionTuple:
    # assert left.op is MathOperator.ADD
    if _add_add_lsimplifiable(left, right):
        v = left + right.left
        if isinstance(v, ExpressionTrait):
            return add_expression(v, right.right)
        expr = v + right.right
        if isinstance(expr, float | int | SymbolTrait | FunctionTrait | ScaledTrait):
            return ExpressionTuple(1, MathOperator.MUL, expr)
        return ExpressionTuple(expr.left, expr.op, expr.right)
    if _add_add_rsimplifiable(left, right):
        v = left + right.right
        if isinstance(v, ExpressionTrait):
            return add_expression(v, right.left)
        expr = v + right.left
        if isinstance(expr, float | int | SymbolTrait | FunctionTrait | ScaledTrait):
            return ExpressionTuple(expr, right.op, right.left)
        return ExpressionTuple(expr.left, expr.op, expr.right)
    return ExpressionTuple(left, MathOperator.ADD, right)
