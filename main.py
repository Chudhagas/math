from abc import abstractmethod
from typing import Callable, TypeVar, override
import math

# TODO: handle undefined/divbyzero
# TODO: color code logs
# TODO: massive clean up

class Token():
    @abstractmethod
    def out(self) -> str:
        pass

A = TypeVar("A", bound=Token)

def combine(*args: dict[str, A]) -> dict[str, A]:
    result: dict[str, A] = {}
    for each in args:
        for name, a in each.items():
            result[name] = a
    return result

class Literal(Token):
    def __init__(self, number: float):
        self._number: float = number

    @property
    def raw(self) -> float:
        return self._number

    @property
    def number(self) -> float:
        val = self._number
        a = val * 1e10
        a = round(a)
        a = a / 1e10
        # print(val, a)
        return a

    @number.setter
    def number(self, val: float):
        self._number = val

    @override
    def out(self):
        return str(self.number)

    @override
    def __str__(self) -> str:
        return f"Literal({self.number})"

    @override
    def __repr__(self) -> str:
        return self.__str__()

Operation = Callable[[Literal, Literal], Literal]
Action = Callable[[Literal], Literal]

class Expression(Token):
    def __init__(self, values: list[Token]) -> None:
        self.values: list[Token] = values

    def evaluate(self):
        tokens = createExprs(self.values.copy())
        # print(f"createExprs:   {tokens}")
        tokens = evalExprs(tokens)
        # print(f"evalExprs:     {tokens}")
        tokens = evalFunctions(tokens)
        # print(f"evalFunctions: {tokens}")
        tokens = negations(tokens)
        # print(f"negations:     {tokens}")
        iter = 0
        while len(tokens) != 1 and iter < 100:
            tokens = combineLiterals(tokens)
            iter += 1
        return tokens

    @override
    def __str__(self) -> str:
        return f"Expr( {self.values} )"

    @override
    def __repr__(self) -> str:
        return self.__str__()

    @override
    def out(self):
        return ' '.join([x.out() for x in self.values])


class Op(Token):
    def __init__(self, symbol: str, operation: Operation, weight: int) -> None:
        self.symbol: str = symbol
        self.operation: Operation = operation
        self.weight: int = weight

    def __call__(self, left: Literal, right: Literal) -> Literal:
        return self.operation(left, right)

    def dump(self) -> str:
        return f"Op({self.symbol}, {self.weight})"

    @override
    def __str__(self) -> str:
        return f"Op({self.symbol})"

    @override
    def __repr__(self) -> str:
        return self.__str__()

    @override
    def out(self):
        return str(self.symbol)


class Function(Token):
    def __init__(self, name: str, action: Action) -> None:
        self.name: str = name
        self.negate: bool = False
        self.action: Action = action

    def __call__(self, value: Literal) -> Literal:
        a: Literal = self.action(value)
        a.number *= -1 if self.negate else 1
        return a

    @override
    def __str__(self) -> str:
        return f"Function({self.name})"

    @override
    def __repr__(self) -> str:
        return self.__str__()

    @override
    def out(self):
        return str(self.name)


operators = {
        "Add": Op('+', lambda x, y: Literal(x.number + y.number), 1),
        "Sub": Op('-', lambda x, y: Literal(x.number - y.number), 1),
        "Mul": Op('*', lambda x, y: Literal(x.number * y.number), 2),
        "Div": Op('/', lambda x, y: Literal(x.number / y.number), 2),
        "+": Op('+', lambda x, y: Literal(x.number + y.number), 1),
        "-": Op('-', lambda x, y: Literal(x.number - y.number), 1),
        "*": Op('*', lambda x, y: Literal(x.number * y.number), 2),
        "/": Op('/', lambda x, y: Literal(x.number / y.number), 2),
        }

functions = {
        "sin": Function("sin", lambda x: Literal(math.sin(x.number))),
        "cos": Function("cos", lambda x: Literal(math.cos(x.number))),
        "tan": Function("tan", lambda x: Literal(math.tan(x.number))),
        "csc": Function("csc", lambda x: Literal(1/math.sin(x.number))),
        "sec": Function("sec", lambda x: Literal(1/math.cos(x.number))),
        "cot": Function("cot", lambda x: Literal(1/math.tan(x.number))),
        "sqrt": Function("sqrt", lambda x: Literal(math.sqrt(x.number))),
        }

constants = {
        "pi": Literal(math.pi)
        }

identifiers = combine(functions, constants) # pyright: ignore[reportArgumentType]


class Negate(Token): # can use similar system to do factorial
    def __init__(self) -> None:
        pass

    @override
    def __str__(self) -> str:
        return f"Negate"

    @override
    def __repr__(self) -> str:
        return self.__str__()

    @override
    def out(self):
        return '-'


class OpenParen(Token):
    def __init__(self) -> None:
        pass

    @override
    def __str__(self) -> str:
        return f"("

    @override
    def __repr__(self) -> str:
        return self.__str__()
    
    @override
    def out(self):
        return '('


class CloseParen(Token):
    def __init__(self) -> None:
        pass

    @override
    def __str__(self) -> str:
        return f")"

    @override
    def __repr__(self) -> str:
        return self.__str__()

    @override
    def out(self):
        return ')'


def is_op(char: str):
    return char in ['+', '-', '*', '/']


def tokenize(input_string):
    ls = [x for x in input_string]
    tokens: list[Token] = []
    count = len(ls)

    i = 0
    haseat = False

    def eat():
        nonlocal i, haseat
        ch = ls[i]
        i += 1
        haseat = True
        return ch

    while i < count:
        haseat = False
        char = ls[i]

        if char == '':
            char = eat()
            continue

        elif char.isdigit():
            numstr = ""
            has_dec = False
            while i < count and (ls[i].isdigit() or (ls[i] == '.' and not has_dec)):
                char = eat()
                if char == '.':
                    has_dec = True
                numstr += char
            tokens.append(Literal(float(numstr)))

        elif is_op(char):
            lop = eat()
            if lop == '-' and i < count: 
                if len(tokens) < 1:
                    tokens.append(Negate())
                elif isinstance(tokens[len(tokens)-1], Op): # and prev is op
                    tokens.append(Negate())
                else:
                    tokens.append(operators[lop])
            else:
                tokens.append(operators[lop])

        elif char.isalpha():
            string = ""
            while i < count and (ls[i].isalpha()):
                char = eat()
                string += char
            if string in identifiers.keys():
                tokens.append(identifiers[string])
            else:
                raise SyntaxError(f"Unknown identifier: {string}")

        elif char == '(':
            char = eat()
            tokens.append(OpenParen())

        elif char == ')':
            char = eat()
            tokens.append(CloseParen())

        if not haseat:
            i += 1

    return tokens


def createExprs(tokens: list[Token]):
    count = len(tokens)
    if count == 0:
        return tokens

    i = 0
    token: Token = tokens[i]

    haseat = False

    def taste(a: type[Token]):
        return isinstance(tokens[i], a)

    def eat():
        nonlocal i, haseat
        token = tokens[i]
        i += 1 # technically the next index
        haseat = True
        return token

    while i < count:
        haseat = False
        token = tokens[i]

        if taste(OpenParen):
            token = eat()
            inner: list[Token] = []
            start = i
            # In the scenario where we open new parenthesis, we need to make sure we include them in the
            # expression and not prematurely exit the loop.
            count_opened = 0
            while i < count: # consume inner
                if taste(OpenParen):
                    count_opened += 1

                elif taste(CloseParen):
                    if count_opened > 0:
                        count_opened -= 1
                    else:
                        break
                token = eat()
                inner.append(token)

            if i < count and taste(CloseParen): # incase we reached end of input before closing parenthesis (missing)
                end = i
                token = eat()
                bef_ind = start - 2
                is_func = bef_ind >= 0 and isinstance(tokens[bef_ind], Function)

                x = start if is_func else start-1
                end = end-1 if is_func else end

                for _ in range(end+1 - x):
                    _ = tokens.pop(x)
                    i -= 1 # necessary!!! for the i < count
                tokens.insert(x, Expression(inner))

                pass
            else:
                raise SyntaxError("Missing closing parenthesis")


        count = len(tokens) # !!!
        if not haseat:
            i+=1

    return tokens
    

def evalExprs(tokens: list[Token]):
    count = len(tokens)
    if count == 0:
        return tokens

    i = 0
    token: Token = tokens[i]

    haseat = False

    def taste(a: type[Token]):
        return isinstance(tokens[i], a)

    def eat():
        nonlocal i, haseat
        token = tokens[i]
        i += 1 # technically the next index
        haseat = True
        return token

    while i < count:
        haseat = False
        token = tokens[i]


        if taste(Expression):
            start = i
            expr: Expression = eat() # pyright: ignore[reportAssignmentType]
            
            tkns = expr.evaluate()
            _ = tokens.pop(start)

            # We must decrement i now because we just removed an item from the list
            i -= 1

            for x in range(len(tkns)):
                tokens.insert(start+x, tkns[x])
                i += 1
            # print(f"{expr} -> {tkns}")


        count = len(tokens)
        if not haseat:
            i+=1

    return tokens


def evalFunctions(tokens: list[Token]):
    count = len(tokens)
    if count == 0:
        return tokens

    i = 0
    token: Token = tokens[i]

    haseat = False

    def taste(a: type[Token]):
        return isinstance(tokens[i], a)

    def eat():
        nonlocal i, haseat
        token = tokens[i]
        i += 1 # the next index we will eat
        haseat = True
        return token

    while i < count:
        haseat = False
        token = tokens[i]


        if taste(Function):
            fstart = i
            token = eat()
            func: Function = token # pyright: ignore[reportAssignmentType]
            if taste(OpenParen):
                token = eat()
                inner: list[Token] = []
                start = i
                while i < count and not taste(CloseParen):
                    token = eat()
                    inner.append(token)

                if taste(CloseParen):
                    fend = i
                    token = eat()
                    if len(inner) == 1:
                        if isinstance(inner[0], Literal):
                            for _ in range(fend+1 - fstart):
                                _ = tokens.pop(fstart)
                                i -= 1
                            tokens.insert(fstart, func.action(inner[0]))
                        else:
                            raise SyntaxError("Function paramaters expression did not return a Literal")
                    elif len(inner) == 0:
                        raise SyntaxError("Function needs parameters")
                    else:
                        raise SyntaxError("Function has too many parameters. Likely a parsing error")

                else:
                    raise SyntaxError("Missing closing parenthesis")
            else:
                raise SyntaxError("Functions need parenthesis")


        count = len(tokens)
        if not haseat:
            i+=1

    return tokens


def negations(tokens: list[Token]):
    count = len(tokens)
    if count == 0:
        return tokens

    i = 0
    token: Token = tokens[i]

    haseat = False

    def taste(a: type[Token]):
        return isinstance(tokens[i], a)

    def eat():
        nonlocal i, haseat
        token = tokens[i]
        i += 1 # technically the next index
        haseat = True
        return token

    while i < count:
        haseat = False
        token = tokens[i]

        if taste(Negate):
            nstart = i
            token = eat()
            if taste(Literal):
                literal: Literal = eat() # pyright: ignore[reportAssignmentType]
                _ = tokens.pop(nstart)
                literal.number *= -1
            else:
                raise SyntaxError("Attempted to negate a non-Literal token. An expression token likely failed evaluation.")

        count = len(tokens)
        if not haseat:
            i+=1

    return tokens


def combineLiterals(tokens: list[Token]):
    count = len(tokens)
    if count == 0:
        return tokens

    i = 0
    token: Token = tokens[i] # pyright: ignore[reportAssignmentType]

    haseat = False

    def taste(a: type[Token]):
        return isinstance(tokens[i], a)

    def eat() -> Token:
        nonlocal i, haseat
        token = tokens[i]
        i += 1 # technically the next index
        haseat = True
        return token

    while i < count:
        haseat = False
        token = tokens[i]

        if taste(Literal):
            if i+1 < count:
                if isinstance(tokens[i+1], Literal):
                    first: Literal = tokens.pop(i) # pyright: ignore[reportAssignmentType]
                    second: Literal = tokens.pop(i) # pyright: ignore[reportAssignmentType]
                    tokens.insert(i, Literal(first.number * second.number))

        count = len(tokens)
        if not haseat:
            i+=1

    i = 0

    # needed for correct order of operations
    for _ in range(len([0 for x in tokens if isinstance(x, Op)])):
        hi = -1
        hw = -1
        ho: Op = None # pyright: ignore[reportAssignmentType]
        for i in range(len(tokens)):
            if isinstance(tokens[i], Op):
                op: Op = tokens[i] # pyright: ignore[reportAssignmentType]
                if hw < op.weight:
                    hi = i
                    hw = op.weight
                    ho = op

        prev_ind = hi-1
        next_ind = hi+1
        if len(tokens) >= 3 and prev_ind >= 0 and next_ind < len(tokens):
            prev = tokens[prev_ind]
            nxt = tokens[next_ind]
            if isinstance(prev, Literal):
                if isinstance(nxt, Literal):
                    left: Literal = prev
                    right: Literal = nxt
                    for _ in range(3):
                        _ = tokens.pop(hi-1)
                    try:
                        val: Literal = ho(left, right)
                    except ZeroDivisionError:
                        raise ZeroDivisionError("You can't divide by zero")
                    # print(f"{ho} {prev} {nxt} {val}")
                    tokens.insert(hi-1, val)
                else:
                    raise SyntaxError(f"Literal expected on the right of operator: '{ho}' '{nxt}' <x-")
            else:
                raise SyntaxError(f"Literal expected on the left of operator: -x>'{prev}' '{ho}'")
        else:
            raise SyntaxError("There must be an expression on both sides of an operator")

    return tokens


