from cheartpy.symbolic.expressions import Expression
from cheartpy.symbolic.trait import ExpressionTrait, FunctionTrait, MathOperator, SymbolTrait


def div_number_to_number(left: float, right: float) -> float:
    return left / right


def div_number_to_symbol(left: float, right: SymbolTrait) -> ExpressionTrait:
    return Expression(left, MathOperator.DIV, right)


def div_symbol_to_number(left: SymbolTrait, right: float) -> ExpressionTrait: ...


def div_number_to_expression(left: float, right: ExpressionTrait) -> ExpressionTrait: ...


def div_expression_to_number(left: ExpressionTrait, right: float) -> ExpressionTrait: ...


def div_number_to_function(left: float, right: FunctionTrait) -> ExpressionTrait: ...


def div_function_to_number(left: FunctionTrait, right: float) -> ExpressionTrait: ...


def div_symbol_to_symbol(left: SymbolTrait, right: SymbolTrait) -> ExpressionTrait: ...


def div_symbol_to_expression(left: SymbolTrait, right: ExpressionTrait) -> ExpressionTrait: ...


def div_expression_to_symbol(left: ExpressionTrait, right: SymbolTrait) -> ExpressionTrait: ...


def div_symbol_to_function(left: SymbolTrait, right: FunctionTrait) -> ExpressionTrait: ...


def div_function_to_symbol(left: FunctionTrait, right: SymbolTrait) -> ExpressionTrait: ...


def div_expression_to_expression(
    left: ExpressionTrait, right: ExpressionTrait
) -> ExpressionTrait: ...


def div_expression_to_function(left: ExpressionTrait, right: FunctionTrait) -> ExpressionTrait: ...


def div_function_to_expression(left: FunctionTrait, right: ExpressionTrait) -> ExpressionTrait: ...


def div_function_to_function(left: FunctionTrait, right: FunctionTrait) -> ExpressionTrait: ...
