import autograd.numpy as NP
from autograd import grad,elementwise_grad
from scipy.misc import derivative
from sympy import pprint, Symbol
from sympy.functions import log


def f(x):   return NP.log(1/((x**3-4*x+1)**2))

x= 2.0
X = Symbol('X')
d = log(1/((X**3-4*X+1)**2))

def analitical(x): return -((2*(3*x**2-4))/(x**3-4*x+1))

dfdx= grad(f)
print "Solution analitical:", analitical(x)
print "Gradient with Autograd:", dfdx(x)
print "Gradient Scipy Finite differences:",derivative(f,2.0,dx=1e-6)
print "Symbolic solution Sympy:"
print ""
print ""
pprint(d.diff(X))
print d.diff(X,2.0)

