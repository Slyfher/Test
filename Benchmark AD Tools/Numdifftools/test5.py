import numpy as np
import numdifftools as nd



fun = lambda x: np.sin(np.exp(x))/(x)

# STEP 1: trace the function evaluation
a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)



