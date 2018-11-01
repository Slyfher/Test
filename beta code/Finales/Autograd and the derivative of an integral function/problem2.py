import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt


def trapz(y, x):
    d = np.diff(x)
    return np.sum((y[0:-1] + y[1:]) * d / 2)

def f(x):
    a = np.sin(x)
    b = np.cos(x)
    t = np.linspace(a, b, 1000)
    y = np.cosh(t**2)
    return trapz(y, t)

# Here is our derivative!
dfdx = grad(f, 0)
#Here is a graphical comparison of the two:

x = np.linspace(0, 2 * np.pi)

analytical = -np.cosh(np.cos(x)**2) * np.sin(x) - np.cosh(np.sin(x)**2) * np.cos(x)
ad = [dfdx(_x) for _x in x]

plt.plot(x, analytical, label='analytical')
plt.plot(x, ad, 'r--', label='AD')
plt.xlabel('x')
plt.ylabel('df/dx')
plt.legend()
plt.show()