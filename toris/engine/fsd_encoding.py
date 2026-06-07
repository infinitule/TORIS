import numpy as np
from typing import Final

# Numerical constants from Section 9.4
LOG_SALIENCE_SCALE: Final[int] = 2**23
LOG_SALIENCE_EPSILON: Final[float] = 1e-7

def log_salience_encode(kappa: float) -> int:
    """
    Encodes contextual salience kappa into log-salience space L,
    then scales it to a 32-bit integer representation.

    L(kappa) = -log(1 - kappa + epsilon)
    s(R) = int32(L(kappa) * SCALE)

    Args:
        kappa: The contextual salience value in [0, 1].

    Returns:
        The encoded salience as a 32-bit integer.
    """
    # Clamp kappa to [0, 1] to ensure numerical stability
    kappa = max(0.0, min(1.0, kappa))

    # Handle kappa = 0 separately to ensure L(0) = 0
    if kappa == 0:
        return 0

    # Calculate L(kappa)
    # Note: as kappa -> 1, (1 - kappa + epsilon) -> epsilon,
    # and -log(epsilon) becomes a large positive number.
    l_kappa = -np.log(1.0 - kappa + LOG_SALIENCE_EPSILON)

    # Scale and convert to int32
    # We use np.int32 to strictly mimic the 32-bit fixed-point integer requirement
    return int(np.int32(l_kappa * LOG_SALIENCE_SCALE))
