import numpy as np
import matplotlib.pyplot as plt
import autograd.numpy as NP
from autograd import grad
import timeit
import pandas as pd


def f(x):   return np.sin(3*x)*np.log(x)
def f_autograd(x):   return NP.sin(3*x)*NP.log(x) #use numpy for autograd

x = 0.7
h = 1e-7

eq = []
name = []
sol = []
t = []



# analytical derivative
eq.append("sin(3*x)*log(x), x =0.7")
start = timeit.default_timer()
dfdx_a = 3 * np.cos( 3*x)*np.log(x) + np.sin(3*x) / x
stop = timeit.default_timer()
sol.append(dfdx_a)
result = (stop-start)
name.append("Analitical differenciation")
t.append(result)

# finite difference
eq.append("sin(3*x)*log(x), x =0.7")
start = timeit.default_timer()
dfdx_fd = (f(x + h) - f(x))/h
stop = timeit.default_timer()
sol.append(dfdx_fd)
result = (stop-start)
name.append("Finite diferences")
t.append(result)

# central difference
eq.append("sin(3*x)*log(x), x =0.7")
start = timeit.default_timer()
dfdx_cd = (f(x+h)-f(x-h))/(2*h)
stop = timeit.default_timer()
sol.append(dfdx_cd)
result = (stop-start)
name.append("Central diferences")
t.append(result)

# automatic diferenciation
eq.append("sin(3*x)*log(x), x =0.7")
start = timeit.default_timer()
dfdx_autograd = grad(f_autograd)
stop = timeit.default_timer()
sol.append(dfdx_autograd(x))
result = (stop-start)
name.append("Automatic differenciation")
t.append(result)

data_insertion={'Derivate':eq,'Name':name,'Solution':sol,'Time':t}
df=pd.DataFrame(data_insertion)
print df

