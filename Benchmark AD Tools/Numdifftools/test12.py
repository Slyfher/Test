import numpy as np
import numdifftools as nd


fun = lambda x: (x**2)*np.exp(x**5)


a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)




