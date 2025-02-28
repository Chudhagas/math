from abc import abstractmethod
from typing import Callable, TypeVar, override
import math

# TODO: handle undefined/divbyzero
# TODO: color code logs
# TODO: massive clean up

# __debug = True
__debug = False

def debug(*args):
    if __debug:
        print(*args)

class _Token():
    @abstractmethod
    def out(self) -> str:
        pass

Token = TypeVar("Token", bound=_Token)

def combine(*args: dict[str, Token]) -> dict[str, Token]:
    result: dict[str, Token] = {}
    for each in args:
        for name, a in each.items():
            result[name] = a
    return result

class Literal(_Token):
    def __init__(self, number: float):
        self._number: float = number

    @property
    def raw(self) -> float:
        return self._number

    @property
    def number(self) -> float:
        val = self._number
        a = val
        if abs(a%1) < 1e-15 and abs(a%1) > 0:
            a *= 1e12
            a = round(a)
            a = a / 1e12
        return a

    @number.setter
    def number(self, val: float):
        self._number = val

    @property
    def round_number(self) -> float:
        val = self._number
        a = val * 1e12
        a = round(a)
        a = a / 1e12
        return a

    @override
    def out(self):
        return str(self.round_number)

    @override
    def __str__(self) -> str:
        return f"Literal({self.number})"

    @override
    def __repr__(self) -> str:
        return self.__str__()

Operation = Callable[[Literal, Literal], Literal]
Action = Callable[[Literal], Literal]

class Expression(_Token):
    def __init__(self, values: list[_Token]) -> None:
        self.values: list[_Token] = values

    def evaluate(self):
        tokens = createExprs(self.values.copy())
        debug(f"createExprs:   {tokens}")
        tokens = evalExprs(tokens)
        debug(f"evalExprs:     {tokens}")
        tokens = evalFunctions(tokens)
        debug(f"evalFunctions: {tokens}")
        tokens = negations(tokens)
        debug(f"negations:     {tokens}")
        iter = 0
        while len(tokens) != 1 and iter < 100:
            tokens = twoLiterals(tokens)
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


class Op(_Token):
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


class Function(_Token):
    def __init__(self, name: str, action: Action) -> None:
        self.name: str = name
        self.action: Action = action

    def __call__(self, value: Literal) -> Literal:
        a: Literal = self.action(value)
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


class Negate(_Token): # can use similar system to do factorial
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


class OpenParen(_Token):
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


class CloseParen(_Token):
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

def test(x: Literal, y: Literal) -> Literal:
    if y.number == 0:
        raise ZeroDivisionError("You can't divide by zero")
    return Literal(x.number / y.number)

operators = {
        "+": Op('+', lambda x, y: Literal(x.number + y.number), 1),
        "-": Op('-', lambda x, y: Literal(x.number - y.number), 1),
        "*": Op('*', lambda x, y: Literal(x.number * y.number), 2),
        # "/": Op('/', lambda x, y: Literal(x.number / y.number), 2),
        "/": Op('/', test, 2),
        "^": Op('^', lambda x, y: Literal(x.number ** y.number), 3),
        "%": Op('^', lambda x, y: Literal(x.number % y.number), 4),
        }

functions = {
        "sin": Function("sin", lambda x: Literal(math.sin(x.number))),
        "cos": Function("cos", lambda x: Literal(math.cos(x.number))),
        "tan": Function("tan", lambda x: Literal(math.tan(x.number))),
        "csc": Function("csc", lambda x: Literal(1/math.sin(x.number))),
        "sec": Function("sec", lambda x: Literal(1/math.cos(x.number))),
        "cot": Function("cot", lambda x: Literal(1/math.tan(x.number))),
        "sqrt": Function("sqrt", lambda x: Literal(math.sqrt(x.number))),
        "ln": Function("ln", lambda x: Literal(math.log(x.number))),
        "log": Function("log", lambda x: Literal(math.log10(x.number))),
        }

constants = {
        "pi": Literal(math.pi),
        "e": Literal(math.e)
        }

identifiers = combine(functions, constants) # pyright: ignore[reportArgumentType]


def is_op(char: str):
    return char in operators.keys()


class StrState:
    def __init__(self, input_string: str) -> None:
        self.ls: list[str] = [x for x in input_string]
        self.char: str
        self.tokens: list[_Token] = []
        self.i: int = 0
        self.has_eaten: bool = False

    def eat(self):
        self.char = self.ls[self.i]
        self.i += 1
        self.has_eaten = True

    def taste(self):
        return self.ls[self.i]

class TokenState:
    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens: list[_Token] = tokens
        self.token: _Token
        self.i: int = 0
        self.has_eaten: bool = False

    def taste(self, a: type[_Token]):
        return isinstance(self.tokens[self.i], a)

    def eat(self):
        self.token = self.tokens[self.i]
        self.i += 1 # technically the next index
        self.has_eaten = True
        return self.token

    def remove_token(self, idx: int):
        _ = self.tokens.pop(idx)
        self.i -= 1


def input_process(func):
    def wrapper(input_string: str)-> list[_Token]:
        state = StrState(input_string)

        while state.i < len(state.ls):
            state.has_eaten = False
            state.char = state.taste()
            if state.char == '':
                state.eat()
                continue
            else:
                func(state)

            if not state.has_eaten:
                state.i += 1

        debug(state.tokens)
        return state.tokens

    return wrapper


def token_process(func):
    def wrapper(tokens: list[_Token]) -> list[_Token]:
        state = TokenState(tokens)

        if len(tokens) == 0:
            return tokens

        while state.i < len(state.tokens):
            state.has_eaten = False
            state.token = state.tokens[state.i]
            func(state)
            if not state.has_eaten:
                state.i += 1

        return tokens

    return wrapper

class TokenizerError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

@input_process
def tokenize(state: StrState):
    eat = state.eat

    if state.char.isdigit():
        numstr = ""
        has_dec = False
        while state.i < len(state.ls) and (state.taste().isdigit() or (state.taste() == '.' and not has_dec)):
            eat()
            if state.char == '.':
                has_dec = True
            numstr += state.char
        state.tokens.append(Literal(float(numstr)))

    elif is_op(state.char):
        eat()
        lop = state.char
        if lop == '-' and state.i < len(state.ls): 
            if len(state.tokens) < 1:
                state.tokens.append(Negate())
            else:
                cur_tok = state.tokens[len(state.tokens)-1]
                if isinstance(cur_tok, Op):
                    state.tokens.append(Negate())
                elif isinstance(cur_tok, OpenParen):
                    state.tokens.append(Negate())
                else:
                    state.tokens.append(operators[lop])
        else:
            state.tokens.append(operators[lop])

    elif state.char.isalpha():
        string = ""
        sI = state.i
        while state.i < len(state.ls) and state.taste().isalpha():
            eat()
            string += state.char
        if string in identifiers.keys():
            state.tokens.append(identifiers[string])
        else:
            raise TokenizerError(f"Unknown identifier", state.ls, sI, state.i-1)

    elif state.char == '(':
        eat()
        state.tokens.append(OpenParen())

    elif state.char == ')':
        eat()
        state.tokens.append(CloseParen())


@token_process
def createExprs(state: TokenState):
    taste = state.taste
    eat = state.eat
    if taste(OpenParen):
        _ = eat()
        inner: list[_Token] = []
        start = state.i
        # In the scenario where we open new parenthesis, we need to make sure we include them in the
        # expression and not prematurely exit the loop.
        count_opened = 0
        while state.i < len(state.tokens): # consume inner
            if taste(OpenParen):
                count_opened += 1

            elif taste(CloseParen):
                if count_opened > 0:
                    count_opened -= 1
                else:
                    break
            _ = eat()
            inner.append(state.token)

        if state.i < len(state.tokens) and taste(CloseParen): # incase we reached end of input before closing parenthesis (missing)
            end = state.i
            _ = eat()
            bef_ind = start - 2
            is_func = bef_ind >= 0 and isinstance(state.tokens[bef_ind], Function)

            x = start if is_func else start-1
            end = end-1 if is_func else end

            for _ in range(end+1 - x):
                state.remove_token(x)
            state.tokens.insert(x, Expression(inner))
        else:
            raise SyntaxError("Missing closing parenthesis", state.tokens, state.i)


@token_process
def evalExprs(state: TokenState):
    taste = state.taste
    eat = state.eat

    if taste(Expression):
        start = state.i
        expr: Expression = eat() # pyright: ignore[reportAssignmentType]
        
        tkns = expr.evaluate()
        state.remove_token(start)

        for x in range(len(tkns)):
            state.tokens.insert(start+x, tkns[x])
            state.i += 1


class FunctionError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)


@token_process
def evalFunctions(state: TokenState):
    taste = state.taste
    eat = state.eat

    if taste(Function):
        fstart = state.i
        token = eat()
        func: Function = token # pyright: ignore[reportAssignmentType]
        if state.i < len(state.tokens) and taste(OpenParen):
            token = eat()
            inner: list[_Token] = []
            while state.i < len(state.tokens) and not taste(CloseParen):
                token = eat()
                inner.append(token)

            if taste(CloseParen):
                fend = state.i
                token = eat()
                if len(inner) == 1:
                    if isinstance(inner[0], Literal):
                        for _ in range(fend+1 - fstart):
                            state.remove_token(fstart)
                        state.tokens.insert(fstart, func.action(inner[0]))
                    else:
                        raise FunctionError("Function parameters expression did not return a Literal", state.tokens, state.i)
                elif len(inner) == 0:
                    raise FunctionError("Function needs parameters", state.tokens, state.i-1)
                else:
                    raise FunctionError("Function has too many parameters. Likely a parsing error", state.tokens, state.i)

            else:
                raise FunctionError("Missing closing parenthesis", state.tokens, state.i-1)
        else:
            raise FunctionError("Functions need parenthesis", state.tokens, state.i)


@token_process
def negations(state: TokenState):
    taste = state.taste
    eat = state.eat
    if taste(Negate):
        nstart = state.i
        _ = eat()
        if taste(Literal):
            literal: Literal = eat() # pyright: ignore[reportAssignmentType]
            _ = state.tokens.pop(nstart)
            literal.number *= -1
        else:
            raise SyntaxError("Attempted to negate a non-Literal token", state.tokens, state.i)


@token_process
def twoLiterals(state: TokenState):
    taste = state.taste
    if taste(Literal):
        if state.i+1 < len(state.tokens):
            if isinstance(state.tokens[state.i+1], Literal):
                first: Literal = state.tokens.pop(state.i) # pyright: ignore[reportAssignmentType]
                second: Literal = state.tokens.pop(state.i) # pyright: ignore[reportAssignmentType]
                state.tokens.insert(state.i, Literal(first.number * second.number))


def combineLiterals(tokens: list[_Token]):
    # needed for correct order of operations
    start_tokens = tokens.copy()
    for j in range(len([0 for x in tokens if isinstance(x, Op)])):
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
                    except ZeroDivisionError as e:
                        raise ZeroDivisionError(*e.args, start_tokens, hi + j*2)
                    tokens.insert(hi-1, val)
                else:
                    raise SyntaxError(f"Literal expected on the right of operator", start_tokens, hi + j*2)
                    # raise SyntaxError(f"Literal expected on the right of operator: '{ho}' '{nxt}' <x-")
            else:
                raise SyntaxError(f"Literal expected on the left of operator", start_tokens, hi + j*2)
                # raise SyntaxError(f"Literal expected on the left of operator: -x>'{prev}' '{ho}'")
        else:
            raise SyntaxError("There must be an expression on both sides of an operator", start_tokens, hi + j*2)
            # raise SyntaxError("There must be an expression on both sides of an operator")

    return tokens


def interpret(inp: str) -> list[_Token]:
    try:
        tkns = tokenize(inp)
        expr = Expression(tkns)
        out = expr.evaluate()
    except ZeroDivisionError as e:
        msg, tks, idx = e.args
        print(f"{msg}: {' '.join([x.out() for x in e.args[1]])}")
        d = sum([len(tks[i].out()) for i in range(len(tks)) if i < idx])# + len(e.args[1])
        print(f"{' ' * (len(msg) + 2 + d + idx)}^")
    except SyntaxError as e:
        msg, tks, idx = e.args
        print(f"{msg}: {' '.join([x.out() for x in e.args[1]])}")
        d = sum([len(tks[i].out()) for i in range(len(tks)) if i < idx])# + len(e.args[1])
        print(f"{' ' * (len(msg) + 2 + d + idx)}^")
    except TokenizerError as e:
        msg, tks, idx, edx = e.args
        print(f"{msg}: {''.join([x for x in e.args[1]])}")
        d = sum([len(tks[i]) for i in range(len(tks)) if i < idx])# + len(e.args[1])
        if idx == edx:
            print(f"{' ' * (len(msg) + 2 + d)}^")
        else:
            print(f"{' ' * (len(msg) + 2 + d)}{'^' * (edx-idx+1)}")
    except FunctionError as e:
        msg, tks, idx = e.args
        print(f"{msg}: {' '.join([x.out() for x in e.args[1]])}")
        d = sum([len(tks[i].out()) for i in range(len(tks)) if i < idx])# + len(e.args[1])
        print(f"{' ' * (len(msg) + 2 + d + idx)}^")
    else:
        return out
    return []
