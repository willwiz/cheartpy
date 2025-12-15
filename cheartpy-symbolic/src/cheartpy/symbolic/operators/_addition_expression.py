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


def add_expression_to_negative(left: ExpressionTrait, right: NegativeTrait) -> ExpressionTrait: ...
def add_expression_to_function(left: ExpressionTrait, right: FunctionTrait) -> ExpressionTrait: ...
def add_expression_to_symbol(left: ExpressionTrait, right: SymbolTrait) -> ExpressionTrait: ...
def add_expression_to_number(left: ExpressionTrait, right: float) -> ExpressionTrait: ...


def add_negative_to_expression(left: NegativeTrait, right: ExpressionTrait) -> ExpressionTrait: ...
def add_function_to_expression(left: FunctionTrait, right: ExpressionTrait) -> ExpressionTrait: ...
def add_symbol_to_expression(left: SymbolTrait, right: ExpressionTrait) -> ExpressionTrait: ...
def add_symbol_to_expression_add_(left: SymbolTrait, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.ADD:
        msg = (
            "add_symbol_to_expression can only handle addition expressions. "
            "Use add_symbol_to_immutable instead."
        )
        raise ValueError(msg)
    if right.left == left:
        return Expression(2 * left, MathOperator.ADD, right.right)
    if right.right == left:
        return Expression(right.left, MathOperator.ADD, 2 * left)
    return Expression(left, MathOperator.ADD, right)


def add_symbol_to_expression_sub_(left: SymbolTrait, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.SUB:
        msg = (
            "add_symbol_to_expression_sub_ can only handle subtraction expressions. "
            "Use add_symbol_to_expression instead."
        )
        raise ValueError(msg)
    if right.left == left:
        return Expression(2 * left, MathOperator.SUB, right.right)
    if right.right == left:
        return Expression(right.left, MathOperator.SUB, 2 * left)
    return Expression(left, MathOperator.ADD, right)


def add_symbol_to_expression_mul_(left: SymbolTrait, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.MUL:
        msg = (
            "add_symbol_to_expression_mul_ can only handle multiplication expressions. "
            "Use add_symbol_to_expression instead."
        )
        raise ValueError(msg)
    if left == right.left:
        return Expression(2 * left, MathOperator.MUL, right.right)
    if left == right.right:
        return Expression(right.left, MathOperator.MUL, 2 * left)
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression(left: float, right: ExpressionTrait) -> ExpressionTrait: ...
def add_number_to_expression_add_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.ADD:
        msg = (
            "add_number_to_expression_add_ can only handle addition expressions. "
            "Use add_number_to_expression instead."
        )
        raise ValueError(msg)
    if isinstance(right.left, float):
        return Expression(left + right.left, right.op, right.right)
    if isinstance(right.right, float):
        return Expression(right.left, right.op, left + right.right)
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression_sub_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.SUB:
        msg = (
            "add_number_to_expression_sub_ can only handle subtraction expressions. "
            "Use add_number_to_expression instead."
        )
        raise ValueError(msg)
    if isinstance(right.left, float):
        return Expression(left - right.left, right.op, right.right)
    if isinstance(right.right, float):
        return Expression(right.left, right.op, right.right - left)
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression_mul_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.MUL:
        msg = (
            "add_number_to_expression_mul_ can only handle multiplication expressions. "
            "Use add_number_to_expression instead."
        )
        raise ValueError(msg)
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression_div_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.DIV:
        msg = (
            "add_number_to_expression_div_ can only handle division expressions. "
            "Use add_number_to_expression instead."
        )
        raise ValueError(msg)
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression_mod_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.MOD:
        msg = (
            "add_number_to_expression_mod_ can only handle modulus expressions. "
            "Use add_number_to_expression instead."
        )
        raise ValueError(msg)
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression_pow_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    if right.op != MathOperator.POW:
        msg = (
            "add_number_to_expression_pow_ can only handle power expressions. "
            "Use add_number_to_expression instead."
        )
        raise ValueError(msg)
    return Expression(left, MathOperator.ADD, right)


def add_number_to_negative(left: float, right: NegativeTrait) -> float | ExpressionTrait: ...


def add_negative_to_number(left: NegativeTrait, right: float) -> float | ExpressionTrait: ...


def add_expressions(left: ALL_TYPES, right: ALL_TYPES) -> float | ExpressionTrait: ...
