import autograd.numpy as NP
from autograd import grad,elementwise_grad
from scipy.misc import derivative
from sympy import pprint, Symbol
from sympy.functions import cos
from sympy.functions import sin
from sympy.functions import exp
import timeit


def f(x):   
	
	return (NP.exp(NP.sin(x)))/(NP.cos(x))

x= 2.0
X = Symbol('X')
d = (exp(sin(X)))/(cos(X))

def analitical(x): return (NP.exp(NP.sin(x)))*(1+NP.tan(x)*(1/NP.cos(x)))

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

