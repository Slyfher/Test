import numpy as np
import numdifftools as nd



fun = lambda x: (4*x**6+5*x+3)*(np.exp(-x**2+5*x+1))


a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)




