import matplotlib.pyplot as plt
import numpy as np

# Physical parameters (eV and Angstroms)
MASS = 940e6
HBAR_C = 1973.27
D_E = 0.755501
ALPHA = 1.44
R_EQ = 0.131349
PREFACTOR = (HBAR_C**2) / (2.0 * MASS)


def assemble_hamiltonian(n_pts, dr, V_vals):
    # Second-derivative finite difference stencil on interior points
    main_diag = np.full(n_pts, 2.0)
    off_diag = np.full(n_pts - 1, -1.0)
    kinetic = PREFACTOR * (
        np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    ) / (dr**2)
    return kinetic + np.diag(V_vals)


def eval_morse_potential(r_arr, diss=D_E, a=ALPHA, r_min=R_EQ):
    shifted = r_arr - r_min
    return diss * (np.exp(-2.0 * a * shifted) - np.exp(-a * shifted))


def solve_morse_state(r_start, r_end, n_pts, state_idx, a=ALPHA, r_min=R_EQ):
    if n_pts <= 0:
        raise ValueError("interior_points must be positive.")
    if not (0 <= state_idx < n_pts):
        raise ValueError("Invalid state_number.")

    dr = (r_end - r_start) / (n_pts + 1)
    r_inner = np.linspace(r_start + dr, r_end - dr, n_pts)

    V_inner = eval_morse_potential(r_inner, diss=D_E, a=a, r_min=r_min)
    H = assemble_hamiltonian(n_pts, dr, V_inner)

    evals, evecs = np.linalg.eigh(H)
    e_val = evals[state_idx]

    u = evecs[:, state_idx]
    if u[np.argmax(np.abs(u))] < 0:
        u = -u

    norm_factor = np.sqrt(np.sum(np.abs(u) ** 2) * dr)
    u_norm = u / norm_factor
    R_radial = u_norm / r_inner

    return r_inner, R_radial, e_val


if __name__ == "__main__":
    r_min_bound = 0.0
    r_max_bound = 10.0
    grid_pts = 600
    target_state = 0

    r_grid, R_wf, ground_E = solve_morse_state(
        r_min_bound, r_max_bound, grid_pts, target_state, a=ALPHA, r_min=R_EQ
    )

    print("Ground state energy =", ground_E, "eV")
    print("Ground state energy =", format(ground_E / 1e6, ".3e"), "MeV")

    plt.figure(figsize=(8, 5))
    plt.plot(r_grid, R_wf, label=f"E = {ground_E:.5f} eV")
    plt.xlabel("Radial distance r (Å)")
    plt.ylabel("Radial wavefunction R(r)")
    plt.title("Morse potential")
    plt.xlim(0, 10)
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.show()
