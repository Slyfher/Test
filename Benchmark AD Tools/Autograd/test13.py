import autograd.numpy as NP
from autograd import grad,elementwise_grad
from scipy.misc import derivative
from sympy import pprint, Symbol
from sympy.functions import exp
import timeit

#45
def f(x): 
	
	return (5*x**4-3*x**2+2*x)*NP.exp(-3*x**2+x-2)
x= 2.0
X = Symbol('X')
d = (5*X**4-3*X**2+2*X)*exp(-3*X**2+X-2)

def analitical(x): return (-30*x**5+5*x**4+38*x**3-15*x**2-4*x+2)*NP.exp(-3*x**2+x-2)

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

