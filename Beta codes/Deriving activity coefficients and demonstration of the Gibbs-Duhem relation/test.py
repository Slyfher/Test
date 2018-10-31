import autograd.numpy as np
from autograd import grad, jacobian
from ad import adnumber
from ad.admath import * 


A12, A21 = 2.04, 1.5461  # Acetone/water https://en.wikipedia.org/wiki/Margules_activity_model

def GexRT(x, y):
    n = x + y
    x1 = x / n
    x2 = y / n
    return n * x1 * x2 * (A21 * x1 + A12 * x2)



lngamma1 = grad(GexRT)     # dGex/dx
lngamma2 = grad(GexRT, 1)  # dGex/dy

x, y = 1.0, 2.0
n = x + y
x1 = x / n
x2 = y / n


# Evaluate the activity coefficients
print('Gradient with autograd:',lngamma1(x, y), lngamma2(x, y))


# Compare that to these analytically derived activity coefficients
print('Analytical:            ', (A12 + 2 * (A21 - A12) * x1) * x2**2, (A21 + 2 * (A12 - A21) * x2) * x1**2)

# Demonstration of the Gibbs-Duhem rule
dg1 = grad(lngamma1)
dg2 = grad(lngamma2)

n = 1.0 # Choose a basis number of moles
x1 = np.linspace(0, 1)
x2 = 1 - x1
x = n * x1
y = n - x

GD = [_x1 * dg1(_x, _y) + _x2 * dg2(_x, _y)
      for _x1, _x2, _x, _y in zip(x1, x2, x, y)]

print(np.allclose(GD, np.zeros(len(GD))))