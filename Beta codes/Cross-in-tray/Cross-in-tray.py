import autograd.numpy as np
from autograd import grad,elementwise_grad

def Crosstray(x, y): return -0.0001*(np.absolute(np.sin(x)*np.sin(y)*np.exp(np.absolute(100-(np.sqrt(x**2+y**2)/np.pi))))+1)**0.1

dfdx = grad(Crosstray,0)
dfdy = grad(Crosstray,1)
print "gradient", dfdx(1.0,-1.0)
print "gradient", dfdy(1.0,-1.0)

