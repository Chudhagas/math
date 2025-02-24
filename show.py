from main import *

inp = input("Equation: ")

tkns = tokenize(inp)
expr = Expression(tkns)
out = expr.evaluate()

print(f"Answer: {' '.join([x.out() for x in out])}")
