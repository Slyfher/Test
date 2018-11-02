import autograd.numpy as NP
from autograd import grad,elementwise_grad
from scipy.misc import derivative
from sympy import pprint, Symbol
from sympy.functions import cos
from sympy.functions import sin
from sympy.functions import tan


import timeit

#28
def f(x):   
	
	return NP.tan((4*x**4-2*x**2+(7/2)*x**-3+5)**-2)

x= 2.0
X = Symbol('X')
d = tan((4*X**4-2*X**2+(7/2)*X**-3+5)**-2)

def analitical(x): return (-2*(16*x**3-4*x-(21/2)*x**-4))/((4*x**4-2*x**2+(7/2)*x**-3+5)**3)*NP.cos((4*x**4-2*x**2+(7/2)*x**-3+5)**3)**-2

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

