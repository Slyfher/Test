import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from scipy.optimize import fmin_bfgs
from scipy.optimize import fmin_bfgs
from autograd import elementwise_grad
import autograd.numpy as auto_np # Thinly-wrapped numpy
from autograd import grad
import timeit


source = np.array([0., 1.]).reshape(-1, 1)
camera = np.array([1., -1.]).reshape(-1, 1)
interface_y = 0.
t = []
n = []


fig, ax = plt.subplots()
ax.scatter(*source, label='source')
ax.scatter(*camera, label='camera')
ax.hlines(interface_y, 0, 1, label='interface')
ax.legend()

n1, n2 = 1., 1.333

# euclidian distance written as Numpy function
dist = lambda vector: np.sqrt(np.sum(np.square(vector.ravel()))) 

def tof(x):
    """Returns time of flight for ray passing through point (x, 0) on the interface."""
    interface_point = np.array([x, 0.]).reshape(-1, 1)
    return dist(source - interface_point) * n1 + dist(interface_point - camera) * n2

x_candidates = np.linspace(0, 1, num=200)
tofs = tof(x_candidates)

fig, ax = plt.subplots()
ax.plot(x_candidates, tofs, label='time of flights')
ax.legend()
ax.set_xlabel('abscissa of interface refraction point')

norm = mpl.colors.Normalize(vmin=tofs.min(),vmax=tofs.max())
cmap = plt.cm.get_cmap('plasma_r')

fig, ax = plt.subplots()
ax.scatter(*source, label='source')
ax.scatter(*camera, label='camera')
for x in x_candidates:
    interface_point = np.array([x, 0.]).reshape(-1, 1)
    points = np.concatenate((source, interface_point, camera), axis=1).T
    ax.plot(points[:, 0], points[:, 1], color=cmap(norm(tof(x))))

# plotting the minimum time of flight
x = x_candidates[np.argmin(tofs)]
interface_point = np.array([x, 0.]).reshape(-1, 1)
points = np.concatenate((source, interface_point, camera), axis=1).T
ax.plot(points[:, 0], points[:, 1], color='k', label='minimum tof')
ax.legend()


# plotting the colorbar, see https://stackoverflow.com/questions/43805821/matplotlib-add-colorbar-to-non-mappable-object
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, 
             ticks=np.linspace(tofs.min(),tofs.max(),21),
             boundaries=np.linspace(tofs.min(),tofs.max(),21),
            label='time of flight')


fmin_bfgs(tof, x0=(0.))
x_candidates[np.argmin(tofs)]
x
dist_autograd = lambda vector: auto_np.sqrt(auto_np.sum(auto_np.square(vector.ravel()))) 

def tof_autograd(x):
    """Returns time of flight for ray passing through point (x, 0) on the interface.
    Autograd version."""
    interface_point = auto_np.array([x, 0.]).reshape(-1, 1)
    return dist_autograd(source - interface_point) * n1 + dist_autograd(interface_point - camera) * n2

tof_gradient = grad(tof_autograd)

tof_gradient(np.array([1.]))

fmin_bfgs(tof, x0=np.array([0.]), fprime=tof_gradient)

start = timeit.default_timer()
fmin_bfgs(tof, x0=np.array([0.]), disp=0)
stop = timeit.default_timer()
result = (stop-start)
n.append("Time tof without autograd")
t.append(result)
start = timeit.default_timer()
fmin_bfgs(tof, x0=np.array([0.]), fprime=tof_gradient, disp=0)
stop = timeit.default_timer()
result = (stop-start)
n.append("Time tof with autograd")
t.append(result)

camera_parallel = np.concatenate((np.linspace(0, 1, num=5).reshape(1, 1, -1), 
                                  -1 * np.ones((5,)).reshape(1, 1, -1)), axis=0)
camera_parallel.shape
x_parallel = np.ones((5, ), dtype=np.float)
x_parallel.shape

dist_parallel = lambda vector: np.sqrt(np.sum(np.square(vector), axis=0)).ravel()
dist_parallel(x_parallel)

def tof_parallel(x):
    """Returns time of flight for ray passing through point (x, 0) on the interface in parallel."""
    x_parallel = x.reshape(1, 1, -1)
    interface_point = np.concatenate((x_parallel, np.zeros_like(x_parallel)))
    return dist_parallel(source[:, :, np.newaxis] - interface_point) * n1 +            dist_parallel(interface_point - camera_parallel) * n2

tof_parallel(x_parallel)


# Now that we have that, we can rewrite this with autograd.

dist_parallel_autograd = lambda vector: auto_np.sqrt(auto_np.sum(auto_np.square(vector), axis=0)).ravel()


def tof_parallel_autograd(x):
    """Returns time of flight for ray passing through point (x, 0) on the interface in parallel.
    Autograd version."""
    x_parallel = x.reshape(1, 1, -1)
    interface_point = auto_np.concatenate((x_parallel, auto_np.zeros_like(x_parallel)))
    return dist_parallel_autograd(source[:, :, np.newaxis] - interface_point) * n1 +            dist_parallel_autograd(interface_point - camera_parallel) * n2

tof_parallel_autograd(x_parallel)

tof_parallel_gradient = elementwise_grad(tof_parallel_autograd)

tof_parallel_gradient(x_parallel)

def gradient_descent(func, x0, fprime, learning_rate=0.1, iters=50):
    """Gradient descent by hand."""
    x = x0.copy()
    for _ in range(iters):
        x -= learning_rate * fprime(x)
    return x

x_parallel_sol = gradient_descent(tof_parallel_autograd, x_parallel, tof_parallel_gradient)
tofs_parallel = tof_parallel(x_parallel_sol)

norm = mpl.colors.Normalize(vmin=tofs_parallel.min(),vmax=tofs_parallel.max())
cmap = plt.cm.get_cmap('plasma_r')

fig, ax = plt.subplots()
ax.scatter(*source, label='source')
ax.scatter(*camera_parallel, label='camera')
for ind, x in enumerate(x_parallel_sol):
    interface_point = np.array([x, 0.]).reshape(-1, 1)
    camera = camera_parallel[:, :, ind]
    points = np.concatenate((source, interface_point, camera), axis=1).T
    ax.plot(points[:, 0], points[:, 1], color=cmap(norm(tof(x))), alpha=0.3, lw=5)
ax.hlines(interface_y, 0, 1, label='interface')
ax.legend()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, 
             ticks=np.linspace(tofs_parallel.min(),tofs_parallel.max(),21),
             boundaries=np.linspace(tofs_parallel.min(),tofs_parallel.max(),21),
            label='time of flight')

plt.show()
N = 24
camera_parallel = np.concatenate((np.linspace(0, 1, num=N).reshape(1, 1, -1), 
                                  -1 * np.ones((N,)).reshape(1, 1, -1)), axis=0)
camera_parallel.shape
x_parallel = np.ones((N, ), dtype=np.float)
x_parallel.shape
x_parallel_sol = gradient_descent(tof_parallel_autograd, x_parallel, tof_parallel_gradient)
tofs_parallel = tof_parallel(x_parallel_sol)

norm = mpl.colors.Normalize(vmin=tofs_parallel.min(),vmax=tofs_parallel.max())
cmap = plt.cm.get_cmap('plasma_r')


fig, ax = plt.subplots()
ax.scatter(*source, label='source')
ax.scatter(*camera_parallel, label='camera')
for ind, x in enumerate(x_parallel_sol):
    interface_point = np.array([x, 0.]).reshape(-1, 1)
    camera = camera_parallel[:, :, ind]
    points = np.concatenate((source, interface_point, camera), axis=1).T
    ax.plot(points[:, 0], points[:, 1], color=cmap(norm(tof(x))), alpha=0.3, lw=5)
ax.hlines(interface_y, 0, 1, label='interface')
ax.legend()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, 
             ticks=np.linspace(tofs_parallel.min(),tofs_parallel.max(),21),
             boundaries=np.linspace(tofs_parallel.min(),tofs_parallel.max(),21),
            label='time of flight')

plt.show
# We can even vary the locations of these points.

X, Y = np.meshgrid(np.linspace(0., 1., num=50), np.linspace(-1., -0.2, num=10))
camera_parallel = np.concatenate((X.reshape(1, -1), Y.reshape(1, -1)))[:, np.newaxis, :]
camera_parallel += np.random.rand(*camera_parallel.shape) / 10

camera_parallel.shape
x_parallel = np.ones((camera_parallel.shape[2], ), dtype=np.float)
x_parallel.shape
x_parallel_sol = gradient_descent(tof_parallel_autograd, x_parallel, tof_parallel_gradient)
tofs_parallel = tof_parallel(x_parallel_sol)

norm = mpl.colors.Normalize(vmin=tofs_parallel.min(),vmax=tofs_parallel.max())
cmap = plt.cm.get_cmap('plasma_r')

fig, ax = plt.subplots()
ax.scatter(*source, label='source')
ax.scatter(*camera_parallel, marker='.', label='cameras')
for ind, x in enumerate(x_parallel_sol):
    interface_point = np.array([x, 0.]).reshape(-1, 1)
    camera = camera_parallel[:, :, ind]
    points = np.concatenate((source, interface_point, camera), axis=1).T
    ax.plot(points[:, 0], points[:, 1], color=cmap(norm(tof(x))), alpha=0.3, lw=2)
ax.hlines(interface_y, 0, 1, label='interface')
ax.legend()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, 
             ticks=np.linspace(tofs_parallel.min(),tofs_parallel.max(),21),
             boundaries=np.linspace(tofs_parallel.min(),tofs_parallel.max(),21),
            label='time of flight')
plt.show()




start = timeit.default_timer()
gradient_descent(tof_parallel_autograd, x_parallel, tof_parallel_gradient)
stop = timeit.default_timer()
result = (stop-start)
n.append("Time parallel tof with gradient")
t.append(result)


data_insertion={'Algorithm':'Ray Trancing','Optimization':n,'Time':t}
df=pd.DataFrame(data_insertion)

print " "
print df


