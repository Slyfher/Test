import numpy as np
import numdifftools as nd



fun = lambda x: (7+2*x)**3/(x**3+4*x**2+1)


a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)



