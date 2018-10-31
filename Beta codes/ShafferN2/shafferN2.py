import autograd.numpy as np
from autograd import grad,elementwise_grad

def shafferN2(x, y): return 0.5+ ((np.sin((x**2-y**2)))**2)-0.5/((1+0.001*(x**2+y**2))**2)

dfdx = grad(shafferN2,0)
dfdy = grad(shafferN2,1)
print "gradient", dfdx(2.0,20.0)
print "gradient", dfdy(2.0,20.0)