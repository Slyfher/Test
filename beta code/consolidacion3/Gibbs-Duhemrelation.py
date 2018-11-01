'''
reference: http://kitchingroup.cheme.cmu.edu/blog/2017/11/22/More-auto-differentiation-goodness-for-science-and-engineering/

'''

import autograd.numpy as np
from autograd import grad, jacobian


A12, A21 = 2.04, 1.5461  # Acetone/water https://en.wikipedia.org/wiki/Margules_activity_model

def GexRT(n1, n2):
    n = n1 + n2
    x1 = n1 / n
    x2 = n2 / n
    return n * x1 * x2 * (A21 * x1 + A12 * x2)

lngamma1 = grad(GexRT)     # dGex/dn1
lngamma2 = grad(GexRT, 1)  # dGex/dn2

n1, n2 = 1.0, 2.0
n = n1 + n2
x1 = n1 / n
x2 = n2 / n

# Evaluate the activity coefficients
print('AD:         ',lngamma1(n1, n2), lngamma2(n1, n2))


# Compare that to these analytically derived activity coefficients
print('Analytical: ', (A12 + 2 * (A21 - A12) * x1) * x2**2, (A21 + 2 * (A12 - A21) * x2) * x1**2)

# Demonstration of the Gibbs-Duhem rule
dg1 = grad(lngamma1)
dg2 = grad(lngamma2)

n = 1.0 # Choose a basis number of moles
x1 = np.linspace(0, 1)
x2 = 1 - x1
n1 = n * x1
n2 = n - n1

GD = [_x1 * dg1(_n1, _n2) + _x2 * dg2(_n1, _n2)
      for _x1, _x2, _n1, _n2 in zip(x1, x2, n1, n2)]

print(np.allclose(GD, np.zeros(len(GD))))