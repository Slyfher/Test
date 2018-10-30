import numpy as np
import numdifftools as nd



fun = lambda x: (5*x**4-3*x**2+2*x)*np.exp(-3*x**2+x-2)


a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)




