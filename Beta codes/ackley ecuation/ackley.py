import autograd.numpy as np
from autograd import grad,elementwise_grad

def ackley(x, y): return -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2)))-np.exp(0.5*(np.cos(2*np.pi*x)+(np.cos(2*np.pi*y))))+np.exp(1)+20

dfdx = grad(ackley,0)
dfdy = grad(ackley,1)
print "gradient", dfdx(1.0,-1.0)
print "gradient", dfdy(1.0,-1.0)

