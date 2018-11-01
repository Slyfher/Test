
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from autograd import grad
import pandas as pd
import timeit



t = []
n = []

#Method 1

R = 0.08206
Pc = 72.9
Tc = 304.2

a = 27 * R**2 * Tc**2 / (Pc * 64)
b = R * Tc / (8 * Pc)

Tr = 1.1

def objective(V, Pr):
    P = Pr * Pc
    T = Tr * Tc
    return P * (V - b) - (R * T)  +  a / V**2 * (V - b)

start = timeit.default_timer()
Pr_range = np.linspace(0.1, 10)
V = [fsolve(objective, 3, args=(Pr,))[0] for Pr in Pr_range]

T = Tr * Tc
P_range = Pr_range * Pc
Z = P_range * V / (R * T)
stop = timeit.default_timer()
result = (stop-start)
n.append("Time Method 1")
t.append(result)
#print Z

plt.plot(Pr_range, Z)
plt.xlabel('$P_r$')
plt.ylabel('Z')
plt.xlim([0, 10])
plt.ylim([0, 2])
#plt.show()



#Method 2

V0, = fsolve(objective, 3, args=(0.1,))
V0


def dPdV(V):
    return -R * T / (V - b)**2 + 2 * a / V**3

def dVdP(V):
    return 1 / dPdV(V)

dPdPr = Pc

def dVdPr(Pr, V):
    return dVdP(V) * dPdPr


start = timeit.default_timer()
Pr_span = (0.1, 10)
Pr_eval, h = np.linspace(*Pr_span, retstep=True)

sol = solve_ivp(dVdPr, Pr_span, (V0,), dense_output=True, max_step=h)

V = sol.y[0]
P = sol.t * Pc
Z = P * V / (R * T)
stop = timeit.default_timer()
result = (stop-start)
n.append("Time Method 2")
t.append(result)
#print Z
plt.plot(sol.t, Z)
plt.xlabel('$P_r$')
plt.ylabel('Z')
plt.xlim([0, 10])
plt.ylim([0, 2])
#plt.show()

#Method 3



def P(V):
    return R * T / (V - b) - a / V**2

# autograd.grad returns a callable that acts like a function
dPdV = grad(P, 0)

def dVdPr(Pr, V):
    return 1 / dPdV(V) * Pc

start = timeit.default_timer()
sol = solve_ivp(dVdPr,  Pr_span, (V0,), dense_output=True, max_step=h)

V, = sol.y
P = sol.t * Pc
Z = P * V / (R * T)
stop = timeit.default_timer()
result = (stop-start)
n.append("Time method 3A")
t.append(result)
#print Z
plt.plot(sol.t, Z)
plt.xlabel('$P_r$')
plt.ylabel('Z')
plt.xlim([0, 10])
plt.ylim([0, 2])
#plt.show()

data_insertion={'Algorithm':'CompressibilityFactor','Optimization':n,'Time':t}
df=pd.DataFrame(data_insertion)

print " "
print df



