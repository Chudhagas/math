from main import *

inp = input("Equation: ")

out = interpret(inp)
# tkns = tokenize(inp)
# expr = Expression(tkns)
# out = expr.evaluate()

if len(out) > 0:
    print(f"Answer: {' '.join([x.out() for x in out])}")
