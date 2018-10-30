import numpy, algopy
from algopy import exp, cos



def f(x): return cos(exp(x))/x

# STEP 1: trace the function evaluation
a = 2.0
cg = algopy.CGraph()
x = algopy.Function(a)
y = f(x)
cg.trace_off()
cg.independentFunctionList = [x]
cg.dependentFunctionList = [y]

# STEP 2: use the computational graph to evaluate derivatives
print('gradient with algopy =', cg.gradient(a))



