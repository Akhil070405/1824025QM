import matplotlib.pyplot as plt
import numpy as np
from scipy.special import eval_hermite, factorial


def make_hamiltonian(n_pts, dx, V):
    # Tridiagonal kinetic matrix using NumPy diag (hbar = 1, m = 1)
    main_diag = np.full(n_pts, 2.0)
    off_diag = np.full(n_pts - 1, -1.0)
    T = (1.0 / (2.0 * dx**2)) * (
        np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    )
    return T + np.diag(V)


def exact_wavefunction(n, x):
    norm_factor = np.sqrt((2.0**n) * factorial(n) * np.sqrt(np.pi))
    return eval_hermite(n, x) * np.exp(-(x**2) / 2.0) / norm_factor


def solve_qho_matrix(L, n_pts, state):
    dx = (2.0 * L) / (n_pts + 1)
    x_in = np.linspace(-L + dx, L - dx, n_pts)
    x_all = np.linspace(-L, L, n_pts + 2)

    V = 0.5 * (x_in**2)
    H = make_hamiltonian(n_pts, dx, V)

    energies, vecs = np.linalg.eigh(H)
    q_nums = np.arange(1, len(energies) + 1)
    exact_E = q_nums + 0.5

    # Extract and normalize selected eigenstate
    phi = np.concatenate(([0.0], vecs[:, state], [0.0]))
    phi = phi / np.sqrt(np.sum(np.abs(phi) ** 2) * dx)

    ana_phi = exact_wavefunction(state, x_all)

    return x_all, phi, energies, exact_E, ana_phi, q_nums


def shoot_qho_rk4(n, x_bound=6.0, points=1000):
    E = n + 0.5
    x = np.linspace(-x_bound, x_bound, points)
    dx = x[1] - x[0]

    psi = np.zeros(points)
    z = np.zeros(points)
    z[0] = 1.0  # Initial conditions S0 = [0, 1]

    def derivatives(p_val, z_val, x_pos):
        V_pos = 0.5 * (x_pos**2)
        return z_val, -2.0 * (E - V_pos) * p_val

    for i in range(points - 1):
        k1_p, k1_z = derivatives(psi[i], z[i], x[i])
        k2_p, k2_z = derivatives(psi[i] + 0.5 * dx * k1_p, z[i] + 0.5 * dx * k1_z, x[i] + 0.5 * dx)
        k3_p, k3_z = derivatives(psi[i] + 0.5 * dx * k2_p, z[i] + 0.5 * dx * k2_z, x[i] + 0.5 * dx)
        k4_p, k4_z = derivatives(psi[i] + dx * k3_p, z[i] + dx * k3_z, x[i] + dx)

        psi[i + 1] = psi[i] + (dx / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        z[i + 1] = z[i] + (dx / 6.0) * (k1_z + 2 * k2_z + 2 * k3_z + k4_z)

    # Normalize and align sign with analytical peak
    psi_norm = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    if psi_norm[np.argmax(np.abs(psi_norm))] < 0:
        psi_norm = -psi_norm

    ana_phi = exact_wavefunction(n, x)
    return x, psi_norm, ana_phi


# --- Part 1: Finite Difference Matrix Method ---
state_num = 1
x, phi, num_E, exact_E, ana_phi, q_nums = solve_qho_matrix(L=8, n_pts=900, state=state_num)

# Plot 1: Wavefunction comparison
plt.figure()
plt.plot(x, phi, label=f"Wavefunction (n={state_num + 1})", color="blue")
plt.plot(x, ana_phi, label="analytical waveform", color="red", linestyle="--")
plt.xlabel("Position (x)")
plt.ylabel("Wavefunction (ψ)")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Numerical vs Analytical Energy
plt.figure()
plt.plot(q_nums, num_E, label="numerical energy")
plt.plot(q_nums, exact_E, label="analytical energy", linestyle="--")
plt.xlabel("quantum number")
plt.ylabel("energy")
plt.title("Numerical energy v/s Analytical energy")
plt.legend()
plt.grid(True)
plt.show()

# --- Part 2: RK4 Shooting Method ---
n_shoot = int(input("enter the state number: "))
state_label = "ground state" if n_shoot == 0 else f"state = {n_shoot}"

x_sh, phi_sh, ana_sh = shoot_qho_rk4(n=n_shoot, x_bound=6.0, points=1000)

# Plot 3: Shooting Method Wavefunction
plt.figure()
plt.plot(x_sh, phi_sh, label="numerical wavefunction")
plt.plot(x_sh, ana_sh, label="analytical wavefunction", linestyle="--")
plt.title(f"wavefunction for {state_label}")
plt.xlabel("x")
plt.ylabel("wavefunction")
plt.legend()
plt.grid(True)
plt.show()
