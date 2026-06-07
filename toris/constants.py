"""TORIS numerical defaults.

These mirror the values in MATH_SPEC.md Section 8. All are empirically
revisable; every revision must be logged in docs/DEVIATIONS.md.
"""

# --- Surprise metric ΔS weighting (MATH_SPEC §3.2) ---
ALPHA = 0.6  # structural surprise weight
BETA = 0.3  # type surprise weight
GAMMA = 0.1  # strength surprise weight

# --- Thresholds ---
THETA_EPSILON = 0.2  # propagation threshold ε (MATH_SPEC §3.3)
THETA_KAPPA = 0.15  # suppression threshold κ after warp (MATH_SPEC §4.2)
THETA_AMPLIFY = 0.5  # amplification threshold κ (MATH_SPEC §4.2 step 3)
THETA_ADD = 0.4  # new-relator instantiation threshold (MATH_SPEC §5.1)
THETA_STRONG = 0.3  # strengthen threshold (MATH_SPEC §5.1)

# --- Plasticity rates (MATH_SPEC §5.1–5.2) ---
ETA_FAST = 0.1  # fast plasticity learning rate
ETA_DECAY = 0.01  # confirmed-prediction weakening rate
ETA_MED = 0.05  # medium plasticity rate (MATH_SPEC §5.2)
CONFIRM_N = 3  # confirmations before a relator is WEAKENed (N, not tabulated)

# --- Slow plasticity (training analog; the TORIS spec §2, not in MATH_SPEC) ---
ETA_SLOW = 0.01  # slow plasticity rate (long-term baseline EMA)
CONSOLIDATION_THRESHOLD = 0.5  # baseline σ above which a relator is consolidated

# --- Chain uncertainty (MATH_SPEC §6) ---
LAMBDA = 0.5  # surprise decay constant in chain uncertainty

# --- Semantic type distances (MATH_SPEC §3.2) ---
D_TYPE_SAME = 0.0
D_TYPE_SIMILAR = 0.3
D_TYPE_UNRELATED = 0.7
D_TYPE_CONTRA = 1.0

# --- Hypotheticals (MATH_SPEC §6) ---
SIGMA_HYPOTHETICAL = 0.1  # strength assigned to an instantiated connector

# --- Section 9: Fast Surprise Dynamics (TFSA + cyclic wave) ---
B_BITS = 16  # bit depth for log-salience fixed-point encoding (§9.9)
S_TORIS_B16 = 562759  # spec magic for B=16 (§9.2) — superseded, see D-19
# Calibrated magic = round(2^B · 0.0450465), the genuine fast-inverse-sqrt
# correction for the bias-free log-salience encoding. The spec's value carries
# a spurious +B/2 offset that makes the literal pipeline degenerate (D-19).
S_TORIS_B16_CALIBRATED = 2952
DELTA_LOG = 0.001  # log(0) prevention offset (§9.3)

H_EULER = 0.1  # Euler integration step for the cyclic wave (§9.9)
N_WAVE_STEPS = 200  # steps to run before instability check (§9.9); needs ~5τ=5/b steps to converge
INSTABILITY_THRESHOLD = 5.0  # = 1/θ_ε ; max sustained ε above this → unstable (§9.9)
TFSA_SCREEN_THRESHOLD = THETA_EPSILON  # = θ_ε, the fast-screen cutoff (§9.9)
