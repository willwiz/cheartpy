from cheartpy.symbolic.expressions import Expression
from cheartpy.symbolic.trait import ExpressionTrait, FunctionTrait, MathOperator, SymbolTrait


def sub_function_to_immutable(
    left: FunctionTrait, right: FunctionTrait | SymbolTrait | float
) -> ExpressionTrait: ...


def sub_symbol_to_immutable(
    left: SymbolTrait, right: FunctionTrait | SymbolTrait | float
) -> ExpressionTrait: ...


def sub_number_to_function_or_symbol(
    left: float, right: FunctionTrait | SymbolTrait
) -> float | ExpressionTrait:
    return Expression(left, MathOperator.SUB, right)


def sub_number_to_number(left: float, right: float) -> float:
    return left - right


def basic_sub_number_to_(
    left: FunctionTrait | SymbolTrait | float, right: FunctionTrait | SymbolTrait | float
) -> float | ExpressionTrait:
    match left:
        case float() | int():
            if isinstance(right, (float, int)):
                return sub_number_to_number(left, right)
            return sub_number_to_function_or_symbol(left, right)
        case SymbolTrait():
            return sub_symbol_to_immutable(left, right)
        case FunctionTrait():
            return sub_function_to_immutable(left, right)
