import numpy, algopy
from algopy import exp



def f(x): return (5*x**4-3*x**2+2*x)*exp(-3*x**2+x-2)

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



