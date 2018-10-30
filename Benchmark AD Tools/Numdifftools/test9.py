import numpy as np
import numdifftools as nd



fun = lambda x: (np.exp(np.sin(x)))/(np.cos(x))


a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)


