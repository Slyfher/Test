from casadi import *


# Create scalar/matrix symbols
x = MX.sym('x',1)
a = 2.0

# Compose into expressions
y = cos(exp(x))/x

# Sensitivity of expression -> new expression
grad_y = gradient(y,x);

# Create a Function to evaluate expression
f = Function('f',[x],[grad_y])

# Evaluate numerically
grad_y_num = f(a);
print "grad casADI:", grad_y_num