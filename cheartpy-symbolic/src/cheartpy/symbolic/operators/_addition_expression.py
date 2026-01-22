from cheartpy.symbolic.expressions import Expression
from cheartpy.symbolic.trait import (
    ALL_TYPES,
    ExpressionTrait,
    FunctionTrait,
    MathOperator,
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
def add_symbol_to_expression_add_(left: SymbolTrait, right: ExpressionTrait) -> ExpressionTrait:
    # assert right.op is MathOperator.ADD
    if right.left == left:
        return Expression(2 * left, MathOperator.ADD, right.right)
    if right.right == left:
        return Expression(right.left, MathOperator.ADD, 2 * left)
    return Expression(left, MathOperator.ADD, right)


def add_symbol_to_expression_sub_(left: SymbolTrait, right: ExpressionTrait) -> ExpressionTrait:
    # assert right.op is MathOperator.SUB
    if right.left == left:
        return Expression(2 * left, MathOperator.SUB, right.right)
    if right.right == left:
        return Expression(right.left, MathOperator.SUB, 2 * left)
    return Expression(left, MathOperator.ADD, right)


def add_symbol_to_expression_mul_(left: SymbolTrait, right: ExpressionTrait) -> ExpressionTrait:
    # assert right.op is MathOperator.MUL
    if left == right.left:
        return Expression(2 * left, MathOperator.MUL, right.right)
    if left == right.right:
        return Expression(right.left, MathOperator.MUL, 2 * left)
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression(left: float, right: ExpressionTrait) -> ExpressionTrait: ...
def add_number_to_expression_add_(left: float, right: ExpressionTrait) -> ExpressionTrait: ...


def add_number_to_expression_sub_(left: float, right: ExpressionTrait) -> ExpressionTrait: ...


def add_number_to_expression_mul_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    # assert right.op is MathOperator.MUL
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression_div_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    # assert right.op is MathOperator.DIV
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression_mod_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    # assert right.op is MathOperator.MOD
    return Expression(left, MathOperator.ADD, right)


def add_number_to_expression_pow_(left: float, right: ExpressionTrait) -> ExpressionTrait:
    # assert right.op is MathOperator.POW
    return Expression(left, MathOperator.ADD, right)


def add_expressions(left: ALL_TYPES, right: ALL_TYPES) -> float | ExpressionTrait: ...
