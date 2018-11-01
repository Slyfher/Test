import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt

def trapz(y, x):
    d = np.diff(x)
    return np.sum((y[0:-1] + y[1:]) * d / 2)


def phi(alpha):
    x = np.linspace(0, 1, 1000)
    y = alpha / (x**2 + alpha**2)
    return trapz(y, x)


# This is the derivative here!
adphi = grad(phi, 0)
#Now, we can plot the derivatives. I will plot both the analytical and automatic differentiated results.



# results from AD
alpha = np.linspace(0.01, 1)

# The AD derivative function is not vectorized, so we use this list comprehension.
dphidalpha = [adphi(a) for a in alpha]

def analytical_dphi(alpha):
    return -1 / (1 + alpha**2)

plt.plot(alpha, analytical_dphi(alpha), label='analytical')
plt.plot(alpha, dphidalpha, 'r--', label='AD')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$frac{d\phi}{d\alpha}$')
plt.legend()
plt.show()

perr = (analytical_dphi (alpha) - dphidalpha) / analytical_dphi (alpha) * 100
plt.plot (alpha, perr, label = 'analytical' )
plt.xlabel (r'$\alpha$')
plt.ylabel ('%error')
plt.show()
