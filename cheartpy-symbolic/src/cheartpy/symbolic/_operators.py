from .trait import MATH_OPS, IOperator


class Add(IOperator):
    __slots__ = ["op"]
    op: str

    def __init__(self) -> None:
        self.op = "+"


class Sub(IOperator):
    __slots__ = ["op"]
    op: str

    def __init__(self) -> None:
        self.op = "-"


class Mul(IOperator):
    def __init__(self) -> None:
        self.op = "*"


class Div(IOperator):
    def __init__(self) -> None:
        self.op = "/"


class Pow(IOperator):
    def __init__(self) -> None:
        self.op = "**"


class Mod(IOperator):
    def __init__(self) -> None:
        self.op = "%"


class Fun(IOperator):
    def __init__(self, name: MATH_OPS) -> None:
        self.op = name
