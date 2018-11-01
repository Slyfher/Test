import numpy as NP
import autograd.numpy as np  
from autograd import grad
import pandas as pd
import timeit



t = []
e = []
n = []

# Compute f(x,y,z) = (2*x+y)*z
x = 1.
y = 2.
z = 3.

start = timeit.default_timer()
# Forward pass
q = 2.*x + y   # Node 1
f = q*z        # Node 2

# Backward pass
df_dq = z          # Node 2 input
df_dz = q          # Node 2 input
df_dx = 2 * df_dq  # Node 1 input
df_dy = 1 * df_dq  # Node 1 input

gradient = NP.array([df_dx, df_dy, df_dz])

stop = timeit.default_timer()
result = (stop-start)
n.append("Manual gradient")
e.append("f(x,y,z) = (2*x+y)*z")
t.append(result)




# ### Autograd



def f(args):
    x,y,z = args
    return (2*x + y)*z

f_grad = grad(f) 

x = 1.
y = 2.
z = 3.

f([x, y, z])
start = timeit.default_timer()
f_grad([x, y, z])
stop = timeit.default_timer()
result = (stop-start)
n.append("Autograd gradient")
e.append("f(x,y,z) = (2*x+y)*z")
t.append(result)



# # Backprop Example 2

# ### Manual backprop



# f(x) = 10*np.exp(np.sin(x)) + np.cos(x)**2

start = timeit.default_timer()
# Forward pass
x = 2
a = NP.sin(x)   # Node 1
b = NP.cos(x)   # Node 1
c = b**2        # Node 3
d = NP.exp(a)   # Node 4
f = 10*d + c    # Node 5 (final output)

# Backward pass
df_dd = 10                    # Node 5 input
df_dc = 1                     # Node 5 input
df_da = NP.exp(a) * df_dd     # Node 4 input
df_db = 2*b * df_dc           # Node 3 input
df_dx =  NP.cos(x) * df_da - NP.sin(x) * df_db  # Node 2 and 1 input
stop = timeit.default_timer()
result = (stop-start)
n.append("Manual gradient")
e.append("f(x) = 10*np.exp(np.sin(x)) + np.cos(x)**2")
t.append(result)


# ### Autograd


def f(x):
    return 10*np.exp(np.sin(x)) + np.cos(x)**2

f_grad = grad(f)

x = 2.

f(x)
start = timeit.default_timer()
f_grad(x)
stop = timeit.default_timer()
result = (stop-start)
n.append("Autograd gradient")
e.append("f(x) = 10*np.exp(np.sin(x)) + np.cos(x)**2")
t.append(result)

data_insertion={'Algorithm':n,'Equation':e,'Time':t}
df=pd.DataFrame(data_insertion)

print " "
print df
