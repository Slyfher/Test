from ad import adnumber
from ad.admath import * 


E0, B0, BP, V0 = -56.466,   0.49,    4.753,  16.573

def Murnaghan(vol):
    E = E0 + B0 * vol / BP * (((V0 / vol)**BP) / (BP - 1.0) + 1.0) - V0 * B0 / (BP - 1.)
    return E

def P(vol):
    vol = adnumber(vol)
    z = Murnaghan(vol)
    dEdV = z.d(vol)
    return -dEdV * 160.21773  # in Gpa



print(P(V0*0.99)) # Pressure at the minimum
print(P(0.99 * V0))  # Compressed