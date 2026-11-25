import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def solve_infinite_well(L, N=1000, k=10):
    dx = L / (N + 1)
    x = np.linspace(0, L, N + 2)

    # Finite-difference Hamiltonian (hbar^2 / 2m = 1)
    t = 1.0 / (2.0 * dx**2)
    H = diags([-t, 2 * t, -t], [-1, 0, 1], shape=(N, N))

    # Solve for lowest k states
    energies, vecs = eigsh(H, k=k, which="SM")
    order = np.argsort(energies)
    energies = energies[order]
    vecs = vecs[:, order]

    # Analytical energies
    n_vals = np.arange(1, k + 1)
    exact_energies = (n_vals**2 * np.pi**2) / (2.0 * L**2)

    # Normalize wavefunctions and add zero boundaries
    psis = []
    for i in range(k):
        psi = np.concatenate(([0.0], vecs[:, i], [0.0]))
        psi = psi / np.sqrt(np.sum(psi**2) * dx)
        if psi[1] < 0:
            psi = -psi
        psis.append(psi)

    return x, psis, energies, n_vals, exact_energies


def shoot_infinite_well(L, n=1, points=200):
    h_bar = 1973.0
    m = 1.0
    E = (n**2 * np.pi**2 * h_bar**2) / (2.0 * m * L**2)

    x = np.linspace(0, L, points)
    dx = x[1] - x[0]

    # RK4 integration for psi'' = -(2mE / h_bar^2) * psi
    psi = np.zeros(points)
    z = np.zeros(points)
    z[0] = 1.0
    k2 = 2.0 * m * E / (h_bar**2)

    for i in range(points - 1):
        k1_psi, k1_z = z[i], -k2 * psi[i]
        k2_psi, k2_z = z[i] + 0.5 * dx * k1_z, -k2 * (psi[i] + 0.5 * dx * k1_psi)
        k3_psi, k3_z = z[i] + 0.5 * dx * k2_z, -k2 * (psi[i] + 0.5 * dx * k2_psi)
        k4_psi, k4_z = z[i] + dx * k3_z, -k2 * (psi[i] + dx * k3_psi)

        psi[i + 1] = psi[i] + (dx / 6.0) * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)
        z[i + 1] = z[i] + (dx / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)

    psi = psi / np.sqrt(np.sum(psi**2) * dx)
    exact_psi = np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)

    return x, psi, exact_psi


# --- 1. Finite Difference Method ---
L = 10
x, psis, energies, n_vals, exact_energies = solve_infinite_well(L, N=1000, k=10)

psi_0 = psis[0]
prob_density = psi_0**2
exact_psi_0 = np.sqrt(2.0 / L) * np.sin(np.pi * x / L)

# Plot 1: Ground State Wavefunction
plt.figure(figsize=(9, 5))
plt.plot(x, psi_0, label=f"Numerical (n=1, E={energies[0]:.4f})")
plt.plot(x, exact_psi_0, "--", label="Analytical")
plt.xlabel("Position x")
plt.ylabel(r"$\psi(x)$")
plt.title("1D Infinite Well - Ground State")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Probability Density
plt.figure(figsize=(9, 5))
plt.plot(x, prob_density, label=r"$|\psi_1(x)|^2$")
plt.xlabel("Position x")
plt.ylabel(r"$|\psi(x)|^2$")
plt.title(f"Probability Density (Area = {np.trapezoid(prob_density, dx=x[1]-x[0]):.2f})")
plt.legend()
plt.grid(True)
plt.show()

# Plot 3: Energy vs Quantum Number
plt.figure(figsize=(9, 5))
plt.plot(n_vals, energies, "o-", label="Numerical Energy")
plt.plot(n_vals, exact_energies, "--", label="Analytical Energy")
plt.xlabel("Quantum Number (n)")
plt.ylabel("Energy")
plt.title("Numerical vs Analytical Energy")
plt.legend()
plt.grid(True)
plt.show()

# --- 2. Shooting Method (RK4) ---
x_shoot, psi_shoot, exact_shoot = shoot_infinite_well(L=10, n=1, points=100)

plt.figure(figsize=(9, 5))
plt.plot(x_shoot, psi_shoot, label="Shooting Method (RK4)")
plt.plot(x_shoot, exact_shoot, "--", label="Analytical")
plt.xlabel("Position x")
plt.ylabel(r"$\psi(x)$")
plt.title("Infinite Well via Shooting Method (n=1)")
plt.legend()
plt.grid(True)
plt.show()
