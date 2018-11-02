import numpy as np
import numdifftools as nd

fun = lambda x: (5*x**2+7*x+2)**2/(x**2+6)


a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)



