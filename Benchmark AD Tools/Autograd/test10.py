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
	
	return (4*x**6+5*x+3)*(NP.exp(-x**2+5*x+1))

x= 2.0
X = Symbol('X')
d = (4*X**6+5*X+3)*exp(-X**2+5*X+1)

def analitical(x): return (-8*x**7+20*x**6+24*x**5-10*x**2+19*x+20)*NP.exp(-x**2+5*x+1)

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

