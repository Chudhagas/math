from main import *

import random

funcs = ["sin(!)", "cos(!)", "tan(!)", "csc(!)", "sec(!)", "cot(!)"]
angles = ["0", "kpi/6", "kpi/4", "kpi/3", "kpi/2", "kpi"] # make on for degrees
helper = [1, 6, 4, 3, 2, 1]

while True:
    func_index = random.randint(0, len(funcs)-1)
    angle_index = random.randint(0, len(angles)-1)
    angle = angles[angle_index]
    reference = helper[angle_index]
    k = random.randint(1, reference*2)
    if k == reference or k % reference == 0:
        k = reference + (1 if random.randint(0, 1) == 0 else -1)
    if k == 1:
        k = ''
    func = funcs[func_index]
    eq = f"{func.replace('!', angle.replace('k', str(k)))}"
    inp = input(f"{eq}: ")
    expr = Expression(tokenize(inp))
    out: list[Token] = expr.evaluate()
    lit: Literal = out[0] # pyright: ignore[reportAssignmentType]
    val: float = lit.number
    print(f"Evaluated as: {val}")

    ans = Expression(tokenize(eq)).evaluate()[0].number
    print(f"Answer: {ans}")
    


