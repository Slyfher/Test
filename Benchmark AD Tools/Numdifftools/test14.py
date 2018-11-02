import numpy as np
import numdifftools as nd


fun = lambda x: ((6*x**2+x)**2)*((x**5+x**6)**4)


a = 2.0
dfun = nd.Gradient(fun)
numdiff = dfun(a)

print('gradient with numdifftools =', numdiff)




