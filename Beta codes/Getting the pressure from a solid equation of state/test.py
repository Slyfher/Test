import autograd.numpy as np
from autograd import grad, jacobian
E0, B0, BP, V0 = -56.466,   0.49,    4.753,  16.573

def Murnaghan(vol):
    E = E0 + B0 * vol / BP * (((V0 / vol)**BP) / (BP - 1.0) + 1.0) - V0 * B0 / (BP - 1.)
    return E

def P(vol):
    dEdV = grad(Murnaghan)
    return -dEdV(vol) * 160.21773  # in Gpa

print(P(V0)) # Pressure at the minimum
print(P(0.99 * V0))  # Compressed