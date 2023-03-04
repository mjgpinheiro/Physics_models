import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import spdiags

# Define the domain size and grid spacing
Lx, Ly = 1.0, 1.0
nx, ny = 100, 100
dx, dy = Lx/nx, Ly/ny

# Define the curl x curl operator
def curl_curl(u):
    dudx = np.gradient(u[1], dx, axis=0)
    dudy = np.gradient(u[0], dy, axis=1)
    return np.array([np.gradient(dudy, dx, axis=0) - np.gradient(dudx, dy, axis=1),
                     -np.gradient(dudx, dx, axis=0) - np.gradient(dudy, dy, axis=1)])

# Define the eigenvalue problem
def eigenproblem(kx, ky):
    k2 = kx**2 + ky**2
    A = spdiags([np.ones(nx*ny), -2*np.ones(nx*ny), np.ones(nx*ny)], [-1, 0, 1], nx*ny, nx*ny)/(dx**2)
    B = spdiags([np.ones(nx*ny), np.ones(nx*ny)], [-1, 1], nx*ny, nx*ny)/(2*dx)
    C = spdiags([np.ones(nx*ny), -2*np.ones(nx*ny), np.ones(nx*ny)], [-nx, 0, nx], nx*ny, nx*ny)/(dy**2)
    D = spdiags([np.ones(nx*ny), np.ones(nx*ny)], [-nx, nx], nx*ny, nx*ny)/(2*dy)
    H = A + C + k2*spdiags(np.ones(nx*ny), 0, nx*ny, nx*ny)
    E = spdiags(curl_curl([D.dot(np.ones(nx*ny)), -B.dot(np.ones(nx*ny))])[0], 0, nx*ny, nx*ny)
    return H, E

# Solve for the resonances
n_modes = 10
resonances = []
for i in range(n_modes):
    k_guess = np.pi*np.sqrt((i+0.5)/Lx**2 + (i+0.5)/Ly**2)
    H, E = eigenproblem(k_guess, 0.0)
    w, v = eigsh(H, k=1, M=E, sigma=0.0, which='LM')
    resonances.append(np.sqrt(w[0]))
