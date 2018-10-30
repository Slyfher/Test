import numpy as np
import numdifftools as nd



fun = lambda x: np.log(1/((x**3-4*x+1)**2))


a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)



