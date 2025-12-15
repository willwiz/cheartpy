from cheartpy.symbolic.expressions import Expression
from cheartpy.symbolic.trait import (
    ALL_TYPES,
    ExpressionTrait,
    FunctionTrait,
    MathOperator,
    SymbolTrait,
)


def sub_expression_to_expression(
    left: ExpressionTrait, right: ExpressionTrait
) -> ExpressionTrait: ...


def sub_expression_to_function(left: ExpressionTrait, right: FunctionTrait) -> ExpressionTrait: ...
def sub_expression_to_symbol(left: ExpressionTrait, right: SymbolTrait) -> ExpressionTrait: ...
def sub_expression_to_number(left: ExpressionTrait, right: float) -> ExpressionTrait: ...
def sub_function_to_expression(left: FunctionTrait, right: ExpressionTrait) -> ExpressionTrait: ...
def sub_symbol_to_expression(left: SymbolTrait, right: ExpressionTrait) -> ExpressionTrait: ...
def sub_number_to_expression(left: float, right: ExpressionTrait) -> ExpressionTrait: ...
def sub_number_to_expression_add_(left: float, right: ExpressionTrait) -> ExpressionTrait: ...


def sub_number_to_expression_sub_(left: float, right: ExpressionTrait) -> ExpressionTrait: ...


def sub_number_to_expression_mul_(left: float, right: ExpressionTrait) -> ExpressionTrait: ...


def sub_function_to_immutable(
    left: FunctionTrait, right: FunctionTrait | SymbolTrait | float
) -> ExpressionTrait: ...


def sub_symbol_to_immutable(
    left: SymbolTrait, right: FunctionTrait | SymbolTrait | float
) -> ExpressionTrait: ...


def sub_number_to_function_or_symbol(
    left: float, right: FunctionTrait | SymbolTrait
) -> ExpressionTrait:
    return Expression(left, MathOperator.SUB, right)


def sub_number_to_number(left: float, right: float) -> float:
    return left - right


def sub_number_to_(left: float, right: ALL_TYPES) -> float | ExpressionTrait: ...
