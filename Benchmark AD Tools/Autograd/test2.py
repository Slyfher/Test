import autograd.numpy as NP
from autograd import grad,elementwise_grad
from scipy.misc import derivative
from sympy import pprint, Symbol


def f(x):   return (7+2*x)**3/(x**3+4*x**2+1)

x= 2.0
X = Symbol('X')
d = (7+2*X)**3/(X**3+4*X**2+1)

def analitical(x): return ((7+2*x)**2)*(-13*x**2-56*x+6)/(x**3+4*x**2+1)**2

dfdx= grad(f)
print "Solution analitical:", analitical(x)
print "Gradient with Autograd:", dfdx(x)
print "Gradient Scipy Finite differences:",derivative(f,2.0,dx=1e-6)
print "Symbolic solution Sympy:"
print ""
print ""
pprint(d.diff(X))



