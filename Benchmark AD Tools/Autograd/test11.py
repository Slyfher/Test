import autograd.numpy as NP
from autograd import grad,elementwise_grad
from scipy.misc import derivative
from sympy import pprint, Symbol
from sympy.functions import exp
from sympy.functions import tan
from sympy.functions import cos
from sympy.functions import sin
import timeit

#45
def f(x): 
	
	return (NP.cos(NP.exp(x)))/x

x= 2.0
X = Symbol('X')
d = (cos(exp(X)))/X

def analitical(x): return -(((x*NP.exp(x))*(NP.sin(NP.exp(x)))+NP.cos(NP.exp(x)))/(x**2))

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

