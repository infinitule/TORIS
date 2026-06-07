import numpy as np
from typing import List, Tuple, Final

# Numerical constants from Section 9.4
RK4_STEP_H: Final[float] = 0.1
WAVE_DAMPING_CONSTANTS: Final[int] = 5
DEFAULT_THETA_EPSILON: Final[float] = 0.2

class CyclicSurpriseSystem:
    """
    Implements the continuous surprise dynamics via a cyclic sine system.
    Surprise flows as a damped wave through relational loops.

    Equations: dε_i/dt = sin(κ_{i+1}) - b * ε_i
    """
    def __init__(self, relators_in_loop: List[any], b: float = DEFAULT_THETA_EPSILON):
        """
        Args:
            relators_in_loop: Ordered list of relators forming a cycle.
            b: Damping coefficient (usually the propagation threshold theta_epsilon).
        """
        self.relators = relators_in_loop
        self.b = b

        # Extract kappas for the loop.
        # Note: ε_i is driven by κ_{i+1}.
        # For the last relator ε_n, it is driven by κ_1.
        self.kappa_vec = np.array([r.kappa for r in relators_in_loop], dtype=float)

    def _f(self, epsilon_vec: np.ndarray) -> np.ndarray:
        """
        The system of differential equations: f(t, ε)_i = sin(κ_{i+1}) - b * ε_i
        """
        n = len(epsilon_vec)
        # Shift kappas: [κ_2, κ_3, ..., κ_n, κ_1]
        kappa_next = np.roll(self.kappa_vec, -1)
        return np.sin(kappa_next) - self.b * epsilon_vec

    def rk4_step(self, epsilon_vec: np.ndarray, h: float = RK4_STEP_H) -> np.ndarray:
        """
        Runge-Kutta 4th order integration step.
        """
        k1 = self._f(epsilon_vec)
        k2 = self._f(epsilon_vec + h * k1 / 2)
        k3 = self._f(epsilon_vec + h * k2 / 2)
        k4 = self._f(epsilon_vec + h * k3)

        return epsilon_vec + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def integrate(self, initial_epsilon: np.ndarray) -> np.ndarray:
        """
        Integrate the system until it reaches steady state or hits T_wave.
        T_wave = 5 / b.
        """
        t_wave = WAVE_DAMPING_CONSTANTS / self.b
        steps = int(t_wave / RK4_STEP_H)

        curr_epsilon = initial_epsilon.copy()

        for _ in range(steps):
            # Fixed-point detection: stop if max derivative is very small
            deriv = self._f(curr_epsilon)
            if np.max(np.abs(deriv)) < 1e-6:
                break

            curr_epsilon = self.rk4_step(curr_epsilon)

        return curr_epsilon
