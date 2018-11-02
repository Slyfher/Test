import autograd.numpy as NP
from autograd import grad,elementwise_grad
from scipy.misc import derivative
from sympy import pprint, Symbol
from sympy.functions import cos
from sympy.functions import sin
from sympy.functions import exp
from sympy.functions import log

import timeit

#40
def f(x):   
	
	return (4*x**3+3*x**2)*NP.exp(x**2+7)

x= 2.0
X = Symbol('X')
d = (4*X**3+3*X**2)*exp(X**2+7)

def analitical(x): return (2*x*NP.exp(x**2+7))*(4*x**3+3*x**2+6*x+3)

dfdx= grad(f)
start= timeit.default_timer()
print "Solution analitical:", analitical(x)
stop = timeit.default_timer()
result =(stop-start)
print "Time Analitical solution:", result
start= timeit.default_timer()
print "Gradient with Autograd:", dfdx(x)
stop = timeit.default_timer()
result =(stop-start)
print "Time Autograd solution:", result
start= timeit.default_timer()
print "Gradient Scipy Finite differences:",derivative(f,2.0,dx=1e-6)
stop = timeit.default_timer()
result =(stop-start)
print "Time Finite differences solution:", result
start= timeit.default_timer()
print "Symbolic solution Sympy:"
print ""
print ""
pprint(d.diff(X))
stop = timeit.default_timer()
result =(stop-start)
print "Time Symbolic solution:", result

