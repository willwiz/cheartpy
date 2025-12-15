from cheartpy.symbolic.expressions import Expression
from cheartpy.symbolic.trait import (
    IMMUTABLE_TYPES,
    ExpressionTrait,
    FunctionTrait,
    MathOperator,
    ScaledSymbolTrait,
    SymbolTrait,
)


def sub_function_to_immutable(left: FunctionTrait, right: IMMUTABLE_TYPES) -> ExpressionTrait:
    return Expression(left, MathOperator.SUB, right)


def sub_function_to_number(left: FunctionTrait, right: float) -> ExpressionTrait:
    return Expression(left, MathOperator.SUB, right)


def sub_symbol_to_immutable(left: SymbolTrait, right: IMMUTABLE_TYPES) -> ExpressionTrait:
    return Expression(left, MathOperator.SUB, right)


def sub_symbol_to_number(left: SymbolTrait, right: float) -> ExpressionTrait:
    return Expression(left, MathOperator.SUB, right)


def sub_s_symbol_to_immutable(left: ScaledSymbolTrait, right: IMMUTABLE_TYPES) -> ExpressionTrait:
    return Expression(left, MathOperator.SUB, right)


def sub_s_symbol_to_number(left: ScaledSymbolTrait, right: float) -> ExpressionTrait:
    return Expression(left, MathOperator.SUB, right)


def sub_number_to_immutable(left: float, right: IMMUTABLE_TYPES) -> float | ExpressionTrait:
    return Expression(left, MathOperator.SUB, right)


def sub_number_to_number(left: float, right: float) -> float:
    return left + right


def basic_sub_number_to_(left: float, right: IMMUTABLE_TYPES | float) -> float | ExpressionTrait:
    match left, right:
        case (float() | int(), float() | int()):
            return sub_number_to_number(left, right)
        case (float() | int(), _):
            return sub_number_to_immutable(left, right)


def basic_sub_function_to_(
    left: FunctionTrait, right: IMMUTABLE_TYPES | float
) -> float | ExpressionTrait:
    match left, right:
        case (FunctionTrait(), float() | int()):
            return sub_function_to_number(left, right)
        case (FunctionTrait(), _):
            return sub_function_to_immutable(left, right)


def basic_sub_symbol_to_(
    left: SymbolTrait, right: IMMUTABLE_TYPES | float
) -> float | ExpressionTrait:
    match left, right:
        case (SymbolTrait(), float() | int()):
            return sub_symbol_to_number(left, right)
        case (SymbolTrait(), _):
            return sub_symbol_to_immutable(left, right)


def basic_sub_s_symbol_to_(
    left: ScaledSymbolTrait, right: IMMUTABLE_TYPES | float
) -> float | ExpressionTrait:
    match left, right:
        case (ScaledSymbolTrait(), float() | int()):
            return sub_s_symbol_to_number(left, right)
        case (ScaledSymbolTrait(), _):
            return sub_s_symbol_to_immutable(left, right)


def basic_sub_(
    left: IMMUTABLE_TYPES | float, right: IMMUTABLE_TYPES | float
) -> float | ExpressionTrait:
    match left:
        case float() | int():
            return basic_sub_number_to_(left, right)
        case FunctionTrait():
            return basic_sub_function_to_(left, right)
        case SymbolTrait():
            return basic_sub_symbol_to_(left, right)
        case ScaledSymbolTrait():
            return basic_sub_s_symbol_to_(left, right)
