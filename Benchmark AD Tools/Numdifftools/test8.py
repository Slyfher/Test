import numpy as np
import numdifftools as nd



fun = lambda x: (x**2+3*x+6)**4/(x+1)

# STEP 1: trace the function evaluation
a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)


