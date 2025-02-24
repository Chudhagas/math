# tokens = tokenize(tokens)
# # print(f"first_pass:      {tokens}")
# tokens = createExprs(tokens)
# # print(f"createExprs:     {tokens}")
# tokens = evalExprs(tokens)
# # print(f"evalExprs:       {tokens}")
# tokens = evalFunctions(tokens)
# # print(f"evalFunctions:   {tokens}")
# tokens = negations(tokens)
# # print(f"negations:   {tokens}")
# iter = 0
# while len(tokens) != 1 and iter < 1:
#     tokens = combineLiterals(tokens)
#     print(f"combineLiterals: {tokens}")
#     iter += 1
#
# print(f"final: {' '.join([x.out() for x in tokens])}")

