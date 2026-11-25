import matplotlib.pyplot as plt
import numpy as np

# Constants
hbar = 1973.0
m_e = 0.511e6
eps_0 = 1.0 / (4.0 * np.pi)
e = 3.795
a0 = 4.0 * np.pi * eps_0 * hbar**2 / (m_e * e**2)


def build_hamiltonian(n_pts, dr, V):
    diag_main = np.full(n_pts, 2.0)
    diag_off = np.full(n_pts - 1, -1.0)
    T = (hbar**2 / (2.0 * m_e * dr**2)) * (
        np.diag(diag_main) + np.diag(diag_off, k=1) + np.diag(diag_off, k=-1)
    )
    return T + np.diag(V)


def calc_potential(r, l):
    V_coulomb = -(e**2) / (4.0 * np.pi * eps_0 * r)
    V_centrifugal = (hbar**2 * l * (l + 1)) / (2.0 * m_e * r**2)
    return V_coulomb + V_centrifugal


def solve_hydrogen(r_max, l, n_pts, state):
    dr = r_max / (n_pts + 1)
    r = np.linspace(dr, r_max - dr, n_pts)

    V = calc_potential(r, l)
    H = build_hamiltonian(n_pts, dr, V)

    energies, states = np.linalg.eigh(H)

    u = states[:, state]
    # Match original sign convention (forces largest-amplitude peak to be positive)
    if u[np.argmax(np.abs(u))] < 0:
        u = -u

    # Normalize u(r) and find R(r) = u(r)/r
    u_norm = u / np.sqrt(np.sum(u**2) * dr)
    R = u_norm / r

    E_num = energies[state]
    E_exact = -13.6 / (state + 1) ** 2

    return r / a0, R * (a0**1.5), E_num, E_exact


# Plotting 1s, 2s, and 3s states
r_max = 20.0
n_pts = 1000
l = 0

plt.figure(figsize=(10, 6))

for state in range(3):
    r_scaled, R_scaled, E_num, E_exact = solve_hydrogen(r_max, l, n_pts, state)
    plt.plot(
        r_scaled,
        R_scaled,
        label=f"{state + 1}s, E = {E_num:.2f} eV, exact energy = {E_exact:.2f} eV",
    )

plt.xlabel(r"$r/a_0$")
plt.ylabel(r"$a_0^{3/2}R(r)$")
plt.xlim(0, 10)
plt.legend()
plt.grid(True)
plt.show()
