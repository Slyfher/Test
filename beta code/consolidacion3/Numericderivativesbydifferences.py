'''
numpy has a function called numpy.diff() that is similar to the one found in matlab. 
It calculates the differences between the elements in your list, and returns a list that is one element shorter, 
which makes it unsuitable for plotting the derivative of a function.
Loops in python are pretty slow (relatively speaking) but they are usually trivial to understand. 
In this script we show some simple ways to construct derivative vectors using loops. 
It is implied in these formulas that the data points are equally spaced. 
If they are not evenly spaced, you need a different approach.

'''

import numpy as np
import autograd.numpy as NP #Autograd customizig
from pylab import *
import time
from autograd import elementwise_grad

'''
These are the brainless way to calculate numerical derivatives. They
work well for very smooth data. they are surprisingly fast even up to
10000 points in the vector.
'''

x = np.linspace(0.78,0.79,100)
y = np.sin(x)
xA = NP.linspace(0.78,0.79,100)
def f(x):   return NP.sin(x)
dy_analytical = np.cos(x)
'''
lets use a forward difference method:
that works up until the last point, where there is not
a forward difference to use. there, we use a backward difference.
'''
tf1 = time.time()
dyf = [0.0]*len(x)
for i in range(len(y)-1):
    dyf[i] = (y[i+1] - y[i])/(x[i+1]-x[i])
#set last element by backwards difference
dyf[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])

print(' Forward difference took %f seconds' % (time.time() - tf1))

'''and now a backwards difference'''
tb1 = time.time()
dyb = [0.0]*len(x)
#set first element by forward difference
dyb[0] = (y[0] - y[1])/(x[0] - x[1])
for i in range(1,len(y)):
    dyb[i] = (y[i] - y[i-1])/(x[i]-x[i-1])

print(' Backward difference took %f seconds' % (time.time() - tb1))

'''and now, a centered formula'''
tc1 = time.time()
dyc = [0.0]*len(x)
dyc[0] = (y[0] - y[1])/(x[0] - x[1])
for i in range(1,len(y)-1):
    dyc[i] = (y[i+1] - y[i-1])/(x[i+1]-x[i-1])
dyc[-1] = (y[-1] - y[-2])/(x[-1] - x[-2])

print(' Centered difference took %f seconds' % (time.time() - tc1))

ag1 = time.time()
ad = elementwise_grad(f)
ad1 = ad(xA)
print(' Autograd took %f seconds' % (time.time() - ag1))

plt.plot(x,dy_analytical,label='analytical derivative')
plt.plot(x,dyf,'--',label='forward')
plt.plot(x,dyb,'--',label='backward')
plt.plot(x,dyc,'--',label='centered')
plt.plot(xA,ad1,'--',label='Autograd')

plt.legend(loc='lower left')
plt.savefig('images/simple-diffs.png')