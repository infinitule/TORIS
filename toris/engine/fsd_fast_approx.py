import numpy as np
import struct
from typing import Final

# Numerical constants from Section 9.4
MAGIC_CONSTANT: Final[int] = 0x5f3759df

def tfsa(kappa: float) -> float:
    """
    TORIS Fast Surprise Approximation (TFSA).
    Computes a fast approximation of the surprise potential 1/sqrt(kappa)
    using the Fast Inverse Square Root bit manipulation on kappa's
    IEEE 754 representation.
    """
    # Handle edge cases
    if kappa <= 0:
        return float('inf')

    # Pack kappa as a 32-bit float, then unpack as a 32-bit integer
    # This is the IEEE 754 bit representation
    s_r = struct.unpack('I', struct.pack('f', kappa))[0]

    # The Fast Inverse Square Root magic
    s_inv_r = MAGIC_CONSTANT - (s_r >> 1)

    # Pack back as integer, unpack as float
    epsilon_fast = struct.unpack('f', struct.pack('I', s_inv_r))[0]

    return float(epsilon_fast)

def newton_refine(epsilon_fast: float, kappa: float) -> float:
    """
    One Newton-Raphson iteration to refine the TFSA approximation.
    epsilon_refined = epsilon_fast * (1.5 - 0.5 * kappa * epsilon_fast^2)
    """
    return epsilon_fast * (1.5 - 0.5 * kappa * (epsilon_fast**2))

def validate_tfsa():
    """
    Validates that the TFSA + Newton refinement has < 1% relative error
    compared to the ground truth 1/sqrt(kappa) for kappa in [0.01, 0.99].
    """
    test_kappas = np.linspace(0.01, 0.99, 100)
    max_error = 0.0

    for k in test_kappas:
        ground_truth = 1.0 / np.sqrt(k)
        e_fast = tfsa(k)
        e_refined = newton_refine(e_fast, k)

        rel_error = abs(e_refined - ground_truth) / ground_truth
        max_error = max(max_error, rel_error)

    print(f"TFSA Max Relative Error (1 Newton step): {max_error:.4%}")
    assert max_error < 0.01, f"TFSA error too high: {max_error:.4%}"

if __name__ == "__main__":
    validate_tfsa()
