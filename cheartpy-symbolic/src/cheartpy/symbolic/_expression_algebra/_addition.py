from cheartpy.symbolic.trait import (
    ALL_TYPES,
    ExpressionTrait,
    ExpressionTuple,
    FunctionTrait,
    MathOperator,
)


def add_expression(left: ExpressionTrait, right: ALL_TYPES) -> ExpressionTuple:
    match right:
        case float() | int():
            return _add_expression_to_float_(left, right)
        case FunctionTrait():
            return _add_expression_to_function_(left, right)
        case _:
            raise NotImplementedError


def _add_expression_to_float_(left: ExpressionTrait, right: float) -> ExpressionTuple:
    match left.op:
        case MathOperator.ADD:
            return _add_expression_to_float_add_(left, right)
        case MathOperator.SUB:
            return _add_expression_to_float_sub_(left, right)
        case MathOperator.MUL:
            return _add_expression_to_float_mul_(left, right)
        case MathOperator.DIV:
            return _add_expression_to_float_div_(left, right)
        case MathOperator.POW:
            return _add_expression_to_float_pow_(left, right)
        case MathOperator.MOD:
            return _add_expression_to_float_mod_(left, right)


def _add_expression_to_float_add_(left: ExpressionTrait, right: float) -> ExpressionTuple:
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


def _add_expression_to_float_sub_(left: ExpressionTrait, right: float) -> ExpressionTuple:
    if left.op != MathOperator.SUB:
        msg = (
            "add_expression_to_float_sub_ can only handle subtraction expressions. "
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
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_float_mod_(left: ExpressionTrait, right: float) -> ExpressionTuple:
    if left.op != MathOperator.MOD:
        msg = (
            "add_expression_to_float_mod_ can only handle modulus expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_function_(left: ExpressionTrait, right: FunctionTrait) -> ExpressionTuple:
    match left.op:
        case MathOperator.ADD:
            return _add_expression_to_function_add_(left, right)
        case MathOperator.SUB:
            return _add_expression_to_function_sub_(left, right)
        case MathOperator.MUL:
            return _add_expression_to_function_mul_(left, right)
        case MathOperator.DIV:
            return _add_expression_to_function_div_(left, right)
        case MathOperator.POW:
            return _add_expression_to_function_pow_(left, right)
        case MathOperator.MOD:
            return _add_expression_to_function_mod_(left, right)


def _add_expression_to_function_add_(
    left: ExpressionTrait, right: FunctionTrait
) -> ExpressionTuple:
    if left.op != MathOperator.ADD:
        msg = (
            "add_expression_to_immutable_add_ can only handle addition expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    if isinstance(left.left, FunctionTrait) and left.left == right:
        return ExpressionTuple(left.left + right, left.op, left.right)
    if isinstance(left.right, FunctionTrait) and left.right == right:
        return ExpressionTuple(left.left, left.op, left.right + right)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_function_sub_(
    left: ExpressionTrait, right: FunctionTrait
) -> ExpressionTuple:
    if left.op != MathOperator.SUB:
        msg = (
            "add_expression_to_function_sub_ can only handle subtraction expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    if isinstance(left.left, FunctionTrait) and left.left == right:
        return ExpressionTuple(left.left + right, left.op, left.right)
    if isinstance(left.right, FunctionTrait) and left.right == right:
        return ExpressionTuple(left.left, left.op, left.right + right)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_function_mul_(
    left: ExpressionTrait, right: FunctionTrait
) -> ExpressionTuple:
    if left.op != MathOperator.MUL:
        msg = (
            "add_expression_to_function_mul_ can only handle multiplication expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    if left.left == right:
        return ExpressionTuple(right, left.op, left.right + 1)
    if left.right == right:
        return ExpressionTuple(left.left + 1, left.op, right)
    return ExpressionTuple(left, MathOperator.ADD, right)


def _add_expression_to_function_div_(
    left: ExpressionTrait, right: FunctionTrait
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


def _add_expression_to_function_pow_(
    left: ExpressionTrait, right: FunctionTrait
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


def _add_expression_to_function_mod_(
    left: ExpressionTrait, right: FunctionTrait
) -> ExpressionTuple:
    if left.op != MathOperator.MOD:
        msg = (
            "add_expression_to_function_mod_ can only handle modulus expressions. "
            "Use add_expression instead."
        )
        raise ValueError(msg)
    return ExpressionTuple(left, MathOperator.ADD, right)
