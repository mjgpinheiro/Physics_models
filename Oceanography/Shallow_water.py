import numpy as np
import matplotlib.pyplot as plt

# Define the parameters of the simulation
L = 1    # length of the simulation box
N = 64   # number of grid points in each direction
T = 10   # total simulation time
dt = 0.01 # time step
g = 9.81 # acceleration due to gravity
H = 1    # mean water depth

# Define the initial water surface and velocity field
x, y = np.meshgrid(np.linspace(-L/2, L/2, N), np.linspace(-L/2, L/2, N))
h0 = H + np.exp(-10*(x**2 + y**2))
u0 = np.zeros((2, N, N))

# Define the gradient operator
def grad(f):
    fx = np.gradient(f[0], axis=1)
    fy = np.gradient(f[1], axis=0)
    return np.array([fx, fy])

# Define the Laplacian operator
def laplace(f):
    return np.gradient(np.gradient(f, axis=1), axis=0)[1] + np.gradient(np.gradient(f, axis=0), axis=1)[0]

# Perform the simulation
h = h0
u = u0
for i in range(int(T/dt)):
    h_half = h - dt/2*grad(h*u)
    u_half = u - dt/2*(u*grad(u) + g*grad(h))
    h = h0 + dt*laplace(h_half) - dt*grad(h_half*u_half)
    u = u0 - dt*(u_half*grad(u_half) + g*grad(h_half))

# Plot the final water surface
plt.imshow(h, origin='lower', extent=(-L/2, L/2, -L/2, L/2))
plt.colorbar()
plt.show()
