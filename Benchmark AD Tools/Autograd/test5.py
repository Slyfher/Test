import autograd.numpy as NP
from autograd import grad,elementwise_grad
from scipy.misc import derivative
from sympy import pprint, Symbol
from sympy.functions import exp
from sympy.functions import sin




def f(x):   return (NP.sin(NP.exp(x)))/(x)

x= 2.0
X = Symbol('X')
d = (sin(exp(X)))/(X)

def analitical(x): return (NP.exp(x)*x*NP.cos(NP.exp(x))-NP.sin(NP.exp(x)))/(x**2)

dfdx= grad(f)
print "Solution analitical:", analitical(x)
print "Gradient with Autograd:", dfdx(x)
print "Gradient Scipy Finite differences:",derivative(f,2.0,dx=1e-6)
print "Symbolic solution Sympy:"
print ""
print ""
pprint(d.diff(X))





