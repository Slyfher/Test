import autograd.numpy as np
from autograd import grad, elementwise_grad
import autograd.numpy.random as npr
from autograd.misc.optimizers import adam

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,   # weight matrix
             rs.randn(outsize) * scale)           # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


def swish(x):
    "see https://arxiv.org/pdf/1710.05941.pdf"
    return x / (1.0 + np.exp(-x))


def Ca(params, inputs):
    "Neural network functions"
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = swish(outputs)    
    return outputs

    
# Here is our initial guess of params:
params = init_random_params(0.1, layer_sizes=[1, 8, 1])

# Derivatives
dCadt = elementwise_grad(Ca, 1)

k = 0.23
Ca0 = 2.0
t = np.linspace(0, 10).reshape((-1, 1))

# This is the function we seek to minimize
def objective(params, step):
    # These should all be zero at the solution
    # dCadt = -k * Ca(t)
    zeq = dCadt(params, t) - (-k * Ca(params, t))
    ic = Ca(params, 0) - Ca0
    return np.mean(zeq**2) + ic**2

def callback(params, step, g):
    if step % 1000 == 0:
        print("Iteration {0:3d} objective {1}".format(step,
                                                      objective(params, step)))

params = adam(grad(objective), params,
              step_size=0.001, num_iters=5001, callback=callback) 


tfit = np.linspace(0, 20).reshape(-1, 1)
import matplotlib.pyplot as plt
plt.plot(tfit, Ca(params, tfit), label='soln')
plt.plot(tfit, Ca0 * np.exp(-k * tfit), 'r--', label='analytical soln')
plt.legend()
plt.xlabel('time')
plt.ylabel('$C_A$')
plt.xlim([0, 20])
plt.savefig('nn-ode.png')