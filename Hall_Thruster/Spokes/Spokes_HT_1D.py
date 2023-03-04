##############################################################
# In a Hall thruster, the spokes are a type of instability
#that can occur in the plasma within the discharge channel. These instabilities
#are characterized by the appearance of periodic structures in the plasma, which
#rotate around the axis of the thruster. Spokes can cause fluctuations in the thrust
#produced by the thruster and can even lead to damage or failure of the device.
# #To study spokes in a Hall thruster, we can use a model known as the 1D Quasi-Neutral Hall Thruster Model (QNHM), which describes the behavior of the plasma in the discharge channel. This model is based on the conservation equations for mass, momentum, and energy in the plasma, as well as the Poisson equation for the electric potential.
# cd PHYSICS MODELS
#Here are some Python code snippets to implement the QNHM equations:

import numpy as np
import matplotlib.pyplot as plt

# Define model parameters
m_i = 2.0 * 1.67e-27   # Ion mass (kg)
m_e = 9.11e-31         # Electron mass (kg)
e = 1.6e-19            # Elementary charge (C)
epsilon_0 = 8.85e-12   # Permittivity of free space (F/m)
L = 0.2                # Length of discharge channel (m)
A = 0.01               # Cross-sectional area of discharge channel (m^2)
B0 = 0.1               # Magnetic field strength at channel center (T)
I = 5.0                # Current through discharge channel (A)
V_anode = 100.0        # Anode voltage (V)
T_e = 5.0              # Electron temperature (eV)
T_i = 0.1              # Ion temperature (eV)
gamma = 1.67           # Adiabatic index
k_B = 1.38e-23         # Boltzmann constant

# Define spatial grid
N = 1000    # Number of spatial grid points
dx = L / N  # Grid spacing
x = np.linspace(dx / 2, L - dx / 2, N)

# Initialize variables
rho = np.zeros(N)     # Charge density (C/m^3)
E = np.zeros(N)       # Electric field (V/m)
phi = np.zeros(N)     # Electric potential (V)
n = np.zeros(N)       # Electron density (m^-3)
T = np.zeros(N)       # Electron temperature (eV)
u = np.zeros(N)       # Ion velocity (m/s)
P = np.zeros(N)       # Plasma pressure (Pa)
B_x = np.zeros(N)     # Magnetic field (T)

# Set initial conditions
n[0] = 1.e18           # Electron density at anode (m^-3)
T[0] = T_e            # Electron temperature at anode (eV)
P[0] = 1.e-6          # Plasma pressure (Pa)
phi[0] =V_anode       #
B_x[0] = B0

# Define time parameters
dt = 1.0e-8    # Time step (s)
t_end = 1.0e-6  # End time (s)
t = 0.0      # Initial time (s)

# Define magnetic field profile
B_x = np.zeros(N)
B_x[x < L / 2]  = 1.0  # Uniform magnetic field on left half
B_x[x >= L / 2] = 0.5  # Uniform magnetic field on right half

# Loop over time steps
while t < t_end:
    # Calculate charge density
    rho = e * (n - np.ones(N) * I / A / e)

    # Calculate electric field
    E = np.gradient(phi) / dx

    # Calculate plasma pressure
    P = n * k_B * T

# Set any negative values of u to zero
    u[u < 0] = 0

    # Calculate ion velocity
    u = np.sqrt(2 * e * (V_anode - phi)) - m_i / np.sqrt(2 * e * (V_anode - phi)) * np.sqrt(k_B * T_i / m_i)
#    print(u) # Expected output: 302.1281858784341
    #print(f"The value of u at position {x} is {u}")
    # Calculate ion density
    n_i = rho / e

    # Calculate ion temperature
    T_i = T_e + (gamma - 1) / gamma * m_e / m_i * (u * u / 2 / k_B - 3/2 * T_e)

# Calculate new electron density and electron temperature
#    n[0:] = n[0:] - dt / A * np.gradient(A * n * u, dx)
    n[0:] = n[0:] - dt * np.gradient(n * u, dx) - dt * rho[0:] / (A * e)

#    print(n[0:])
#    n[0:] = n[0:] - dt / A * np.gradient(A * n * u, dx)
#    print(n[0:])
    T[1:-1] = T[1:-1] - dt / (gamma - 1) / np.where(n[1:-1] != 0, n[1:-1], 1) * (np.gradient(P, dx)[1:-1] + (T[2:] - 2 * T[1:-1] + T[:-2]) / dx**2)

#    T = T[0:] - (dt / (gamma - 1)) / n * P * np.gradient(u, dx)
#print(t,T[0])
# Calculate new electric potential
    rho_sum = np.cumsum(rho * dx)
    phi = np.zeros(N)
#    print(phi)
for i in range(1, N):
    phi[1:-1] = phi[1:-1] + dt / epsilon_0 * (rho[1:-1] - (B_x[2:] - B_x[:-2]) / dx / 2 / epsilon_0)

# Advance time
#print(t)
t += dt
plt.figure()
plt.plot(x, n)
plt.xlabel('Position (m)')
plt.ylabel('Electron density (m$^{-3}$)')
plt.show()

plt.figure()
plt.plot(x, T)
plt.xlabel('Position (m)')
plt.ylabel('Electron temperature (eV)')
plt.show()

plt.figure()
plt.plot(x, phi)
plt.xlabel('Position (m)')
plt.ylabel('Electric potential (V)')
plt.show()

#def update_continuity_equation(n, u, S, D, dt, dx):
#    """
#    Update the continuity equation using the Euler method
#    """
#    A = np.sqrt(m_i / (2 * np.pi * e * n))
#    n[1:-1] = n[1:-1] - dt / A * np.gradient(A * n * u, dx) + dt * S[1:-1]
#    n[1:-1] = n[1:-1] - D[1:-1] * n[1:-1] * dt
#    # Apply boundary conditions
#    n[0] = n_i
#    n[-1] = n[-2]

# Xenon source term
#def S_xe(n, T_e):
#    """
#    Computes the source term for Xenon.
#    :param n: number density of Xenon
#    :param T_e: electron temperature
#    :return: source term for Xenon
#    """
    # constants
#    nn = 1.0   #
#    A1 = 8.7e-17  # cm^3/s
#    B1 = 6.5e-9  # cm^3/s/eV
#    E_ion = 12.13  # eV
#
    # compute the source term
#    S_xe = nn * (A1 * np.sqrt(T_e) * np.exp(-E_ion / T_e) - B1 * n)
#
#    return S_xe


### END OF PROGRAM
