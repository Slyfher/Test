import numpy as np
import numdifftools as nd



fun = lambda x: (x**3+4*x**2)*(5*x+4*x**2)**3

# STEP 1: trace the function evaluation
a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)


