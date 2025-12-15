from cheartpy.symbolic.expressions import Expression
from cheartpy.symbolic.trait import (
    ALL_TYPES,
    ExpressionTrait,
    FunctionTrait,
    MathOperator,
    NegativeTrait,
    SymbolTrait,
)


def add_expression_to_expression(
    left: ExpressionTrait, right: ExpressionTrait
) -> ExpressionTrait: ...


def add_expression_to_function(left: ExpressionTrait, right: FunctionTrait) -> ExpressionTrait: ...
def add_expression_to_symbol(left: ExpressionTrait, right: SymbolTrait) -> ExpressionTrait: ...
def add_expression_to_number(left: ExpressionTrait, right: float) -> ExpressionTrait: ...
def add_function_to_expression(left: FunctionTrait, right: ExpressionTrait) -> ExpressionTrait: ...
def add_symbol_to_expression(left: SymbolTrait, right: ExpressionTrait) -> ExpressionTrait: ...
def add_number_to_expression(left: float, right: ExpressionTrait) -> ExpressionTrait: ...
def add_number_to_expression_add_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.ADD:
        msg = (
            "add_number_to_expression_add_ can only handle addition expressions. "
            "Use add_number_to_expression instead."
        )
        raise ValueError(msg)
    if isinstance(right.left, float):
        return Expression(left + right.left, MathOperator.ADD, right.right)
    if isinstance(right.right, float):
        return Expression(right.left, MathOperator.ADD, left + right.right)
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression_sub_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.SUB:
        msg = (
            "add_number_to_expression_sub_ can only handle subtraction expressions. "
            "Use add_number_to_expression instead."
        )
        raise ValueError(msg)
    if isinstance(right.left, float):
        return Expression(left - right.left, MathOperator.SUB, right.right)
    if isinstance(right.right, float):
        return Expression(right.left, MathOperator.SUB, right.right - left)
    return Expression(left, MathOperator.SUB, right)


def add_number_to_expression_mul_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.MUL:
        msg = (
            "add_number_to_expression_mul_ can only handle multiplication expressions. "
            "Use add_number_to_expression instead."
        )
        raise ValueError(msg)
    if isinstance(right.left, float):
        return Expression(left * right.left, MathOperator.MUL, right.right)
    if isinstance(right.right, float):
        return Expression(right.left, MathOperator.MUL, left * right.right)
    return Expression(left, MathOperator.MUL, right)


def add_function_to_immutable(
    left: FunctionTrait, right: FunctionTrait | SymbolTrait | float
) -> ExpressionTrait:
    return Expression(left, MathOperator.ADD, right)


def add_symbol_to_immutable(
    left: SymbolTrait, right: FunctionTrait | SymbolTrait | float
) -> ExpressionTrait:
    return Expression(left, MathOperator.ADD, right)


def add_number_to_function_or_symbol(
    left: float, right: FunctionTrait | SymbolTrait
) -> ExpressionTrait:
    return Expression(left, MathOperator.ADD, right)


def add_number_to_negative(left: float, right: NegativeTrait) -> ExpressionTrait:
    return Expression(left, MathOperator.SUB, right.value)


def add_number_to_number(left: float, right: float) -> float:
    return left + right


def add_number_to_(left: float, right: ALL_TYPES) -> float | ExpressionTrait:
    match right:
        case float() | int():
            return add_number_to_number(left, right)
        case SymbolTrait() | FunctionTrait():
            return add_number_to_function_or_symbol(left, right)
        case ExpressionTrait():
            return add_number_to_expression(left, right)
        case NegativeTrait():
            return add_number_to_negative(left, right)
