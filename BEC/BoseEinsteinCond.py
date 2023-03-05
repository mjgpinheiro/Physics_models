import numpy as np
from scipy.fftpack import fft, ifft

# Define the parameters of the simulation
L = 20    # length of the simulation box
N = 512   # number of grid points
T = 10    # total simulation time
dt = 0.01 # time step
m = 1     # mass of the particles
g = 1     # interaction strength
hbar = 1  # reduced Planck constant

# Define the external potential
def V(x):
    return 0.5 * x**2

# Define the initial wave function
def psi0(x):
    return np.exp(-x**2)

# Define the grid and Fourier space variables
x = np.linspace(-L/2, L/2, N)
dx = x[1] - x[0]
k = 2*np.pi*np.fft.fftfreq(N, dx)
dk = k[1] - k[0]

# Define the Hamiltonian operator
def H(psi):
    psi_hat = fft(psi)
    return ifft(-0.5*hbar**2/m*k**2*psi_hat + V(x)*psi_hat + g*np.abs(psi)**2*psi_hat)

# Perform the simulation
psi = psi0(x)
for i in range(int(T/dt)):
    psi_half = np.exp(-0.5j*dt/H(psi))*psi
    psi = np.exp(-1j*dt/H(psi_half))*psi_half

# Plot the final wave function
import matplotlib.pyplot as plt
plt.plot(x, np.abs(psi)**2)
plt.show()
