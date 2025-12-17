from cheartpy.symbolic.expressions import Expression
from cheartpy.symbolic.scaled import Scaled
from cheartpy.symbolic.trait import (
    ALL_TYPES,
    ExpressionTrait,
    FunctionTrait,
    MathOperator,
    ScaledTrait,
    SymbolTrait,
)


def symbol_add_(left: SymbolTrait, right: ALL_TYPES) -> ALL_TYPES:
    match right:
        case float() | int():
            res = Expression(left, MathOperator.ADD, right)
        case SymbolTrait():
            if left == right:
                res = Scaled(2, left)
            res = Expression(left, MathOperator.ADD, right)
        case FunctionTrait():
            res = Expression(left, MathOperator.ADD, right)
        case ScaledTrait():
            if right.value == left:
                res = Scaled(right.scalar + 1, left)
            res = Expression(left, MathOperator.ADD, right)
        case ExpressionTrait():
            res = symbol_add_expression_(left, right)
    return res


def symbol_add_expression_(left: SymbolTrait, right: ExpressionTrait) -> ALL_TYPES:
    match right.op:
        case MathOperator.ADD:
            res = _symbol_add_expression_add_(left, right)
        case MathOperator.SUB:
            res = _symbol_add_expression_sub_(left, right)
        case MathOperator.MUL:
            raise NotImplementedError
        case MathOperator.DIV:
            raise NotImplementedError
        case MathOperator.POW:
            raise NotImplementedError
        case MathOperator.MOD:
            raise NotImplementedError
    return res


def _symbol_add_expression_add_(left: SymbolTrait, right: ExpressionTrait) -> ALL_TYPES:
    if right.op is not MathOperator.ADD:
        msg = "Right expression must be an addition expression. Use _sym_add_ instead."
        raise ValueError(msg)
    if right.left == left:
        return Expression(Scaled(2, left), right.op, right.right)
    if right.right == left:
        return Expression(right.left, right.op, Scaled(2, left))
    if isinstance(right.left, ScaledTrait) and right.left.value == left:
        return Expression(Scaled(right.left.scalar + 1, left), right.op, right.right)
    if isinstance(right.right, ScaledTrait) and right.right.value == left:
        return Expression(right.left, right.op, Scaled(right.right.scalar + 1, left))
    return Expression(left, MathOperator.ADD, right)


def _symbol_add_expression_sub_(left: SymbolTrait, right: ExpressionTrait) -> ALL_TYPES:
    if right.op is not MathOperator.SUB:
        msg = "Right expression must be a subtraction expression. Use _sym_add_ instead."
        raise ValueError(msg)
    if right.left == left:
        return Expression(Scaled(2, left), right.op, right.right)
    if right.right == left:
        return right.left
    if isinstance(right.left, ScaledTrait) and right.left.value == left:
        return Expression(Scaled(right.left.scalar + 1, left), right.op, right.right)
    if isinstance(right.right, ScaledTrait) and right.right.value == left:
        return Expression(
            right.left,
            right.op,
            Scaled(right.right.scalar - 1, left),
        )
    return Expression(left, MathOperator.ADD, right)
