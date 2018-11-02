import numpy as np
import numdifftools as nd




fun = lambda x: (4*x**3+3*x**2)*np.exp(x**2+7)


a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)





