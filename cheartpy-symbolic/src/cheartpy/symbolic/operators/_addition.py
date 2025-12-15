from cheartpy.symbolic.expressions import Expression
from cheartpy.symbolic.symbols import ScaledSymbol
from cheartpy.symbolic.trait import (
    IMMUTABLE_TYPES,
    ExpressionTrait,
    FunctionTrait,
    MathOperator,
    ScaledSymbolTrait,
    SymbolTrait,
)


def add_function_to_immutable(left: FunctionTrait, right: IMMUTABLE_TYPES) -> ExpressionTrait:
    if isinstance(right, ScaledSymbolTrait) and (right.scale < 0):
        right = ScaledSymbol(-right.scale, right.value)
        return Expression(left, MathOperator.SUB, right)
    return Expression(left, MathOperator.ADD, right)


def add_function_to_number(left: FunctionTrait, right: float) -> ExpressionTrait:
    return Expression(left, MathOperator.ADD, right)


def add_symbol_to_immutable(left: SymbolTrait, right: IMMUTABLE_TYPES) -> ExpressionTrait:
    if isinstance(right, ScaledSymbolTrait) and (right.scale < 0):
        right = ScaledSymbol(-right.scale, right.value)
        return Expression(left, MathOperator.SUB, right)
    return Expression(left, MathOperator.ADD, right)


def add_symbol_to_number(left: SymbolTrait, right: float) -> ExpressionTrait:
    return Expression(left, MathOperator.ADD, right)


def add_s_symbol_to_immutable(left: ScaledSymbolTrait, right: IMMUTABLE_TYPES) -> ExpressionTrait:
    if isinstance(right, ScaledSymbolTrait) and (right.scale < 0):
        right = ScaledSymbol(-right.scale, right.value)
        return Expression(left, MathOperator.SUB, right)
    return Expression(left, MathOperator.ADD, right)


def add_s_symbol_to_number(left: ScaledSymbolTrait, right: float) -> ExpressionTrait:
    return Expression(left, MathOperator.ADD, right)


def add_number_to_immutable(left: float, right: IMMUTABLE_TYPES) -> float | ExpressionTrait:
    if isinstance(right, ScaledSymbolTrait) and (right.scale < 0):
        right = ScaledSymbol(-right.scale, right.value)
        return Expression(left, MathOperator.SUB, right)
    return Expression(left, MathOperator.ADD, right)


def add_number_to_number(left: float, right: float) -> float:
    return left + right


def basic_add_number_to_(left: float, right: IMMUTABLE_TYPES | float) -> float | ExpressionTrait:
    match left, right:
        case (float() | int(), float() | int()):
            return add_number_to_number(left, right)
        case (float() | int(), _):
            return add_number_to_immutable(left, right)


def basic_add_function_to_(
    left: FunctionTrait, right: IMMUTABLE_TYPES | float
) -> float | ExpressionTrait:
    match left, right:
        case (FunctionTrait(), float() | int()):
            return add_function_to_number(left, right)
        case (FunctionTrait(), _):
            return add_function_to_immutable(left, right)


def basic_add_symbol_to_(
    left: SymbolTrait, right: IMMUTABLE_TYPES | float
) -> float | ExpressionTrait:
    match left, right:
        case (SymbolTrait(), float() | int()):
            return add_symbol_to_number(left, right)
        case (SymbolTrait(), _):
            return add_symbol_to_immutable(left, right)


def basic_add_s_symbol_to_(
    left: ScaledSymbolTrait, right: IMMUTABLE_TYPES | float
) -> float | ExpressionTrait:
    match left, right:
        case (ScaledSymbolTrait(), float() | int()):
            return add_s_symbol_to_number(left, right)
        case (ScaledSymbolTrait(), _):
            return add_s_symbol_to_immutable(left, right)


def basic_add_(
    left: IMMUTABLE_TYPES | float, right: IMMUTABLE_TYPES | float
) -> float | ExpressionTrait:
    match left:
        case float() | int():
            return basic_add_number_to_(left, right)
        case FunctionTrait():
            return basic_add_function_to_(left, right)
        case SymbolTrait():
            return basic_add_symbol_to_(left, right)
        case ScaledSymbolTrait():
            return basic_add_s_symbol_to_(left, right)
