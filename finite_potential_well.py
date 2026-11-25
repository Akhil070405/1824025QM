import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal


a = 1.0
V0 = 20.0


L = 6
dx = 0.01

x = np.arange(-L, L + dx/2, dx)


V = np.where(np.abs(x) < a, 0.0, V0)


N = len(x) - 2

diagonal = 2/dx**2 + V[1:-1]

off_diagonal = np.full(N-1, -1/dx**2)


energies, wavefunctions = eigh_tridiagonal(
    diagonal,
    off_diagonal,
    select='i',
    select_range=(0, 5)
)


print("Numerical energy eigenvalues:")

for i, E in enumerate(energies):
    print(f"State {i+1}: {E:.6f}")


psi = np.zeros((len(x), wavefunctions.shape[1]))

psi[1:-1, :] = wavefunctions


for i in range(psi.shape[1]):
    norm = np.sqrt(np.trapezoid(psi[:, i]**2, x))
    psi[:, i] /= norm

plt.figure(figsize=(9, 5))

plt.plot(x, V, label="Potential V(x)")

plt.xlabel("Position (x)")
plt.ylabel("Potential Energy")
plt.title("Finite Potential Well")

plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))

for i in range(3):
    plt.plot(
        x,
        psi[:, i],
        label=f"State {i+1}, E={energies[i]:.3f}"
    )

plt.xlabel("Position (x)")
plt.ylabel("Wavefunction")
plt.title("Numerical Wavefunctions")

plt.grid()
plt.legend()
plt.show()


plt.figure(figsize=(10, 6))

for i in range(3):
    plt.plot(
        x,
        psi[:, i]**2,
        label=f"State {i+1}"
    )

plt.xlabel("Position (x)")
plt.ylabel("Probability Density")
plt.title("Probability Densities")

plt.grid()
plt.legend()
plt.show()
plt.figure(figsize=(9, 5))
plt.plot(x, V, label="Potential V(x)")
plt.xlabel("Position (x)")
plt.ylabel("Potential Energy")
plt.title("Finite Potential Well")
plt.grid()
plt.legend()
plt.show()
