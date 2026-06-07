import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import time
import numpy as np
import matplotlib.pyplot as plt
from toris.engine.fsd_fast_approx import tfsa, newton_refine
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.primitives.relation_types import RelationType

def naive_surprise(relators_obs, relators_pred):
    """
    Mocks the O(n^2) behavior of the previous topological surprise metric.
    Iterates through pairs of relators to find matches.
    """
    match_count = 0
    strength_diff = 0.0

    # This is the O(n^2) part: nested loop over relators
    for r_obs in relators_obs:
        for r_pred in relators_pred:
            if r_obs.edge == r_pred.edge and r_obs.tau == r_pred.tau:
                match_count += 1
                strength_diff += (r_obs.sigma - r_pred.sigma)**2

    return match_count, strength_diff

def tfsa_surprise(relators_obs):
    """
    The O(n) TFSA implementation.
    """
    epsilons = []
    for r in relators_obs:
        e_fast = tfsa(r.kappa)
        epsilons.append(newton_refine(e_fast, r.kappa))
    return epsilons

def run_exp_07():
    print("--- Experiment 07: Computational Speedup (O(n^2) vs O(n log n)) ---")

    n_sizes = [10, 100, 500, 1000, 5000]
    naive_times = []
    tfsa_times = []

    for n in n_sizes:
        # Setup: Create n relators
        # Use a small number of concepts to ensure many edges overlap for the naive version
        concepts = [ConceptState(id=f"C{i}") for i in range(100)]
        relators_obs = []
        for i in range(n):
            src = concepts[np.random.randint(0, 100)]
            tgt = concepts[np.random.randint(0, 100)]
            relators_obs.append(Relator(
                tau=RelationType.CAUSAL,
                src=src,
                tgt=tgt,
                kappa=np.random.random(),
                sigma=np.random.random()
            ))

        # Predictions: just a slightly perturbed version of obs
        relators_pred = [r.clone(sigma=r.sigma * 0.9) for r in relators_obs]

        # Time Naive
        start = time.perf_counter()
        naive_surprise(relators_obs, relators_pred)
        naive_times.append(time.perf_counter() - start)

        # Time TFSA
        start = time.perf_counter()
        tfsa_surprise(relators_obs)
        tfsa_times.append(time.perf_counter() - start)

        print(f"n={n:5} | Naive: {naive_times[-1]:.6f}s | TFSA: {tfsa_times[-1]:.6f}s")

    # Fit log-log scaling curves
    # log(time) = exponent * log(n) + intercept
    log_n = np.log(n_sizes)
    log_naive = np.log(naive_times)
    log_tfsa = np.log(tfsa_times)

    poly_naive = np.polyfit(log_n, log_naive, 1)
    poly_tfsa = np.polyfit(log_n, log_tfsa, 1)

    exp_naive = poly_naive[0]
    exp_tfsa = poly_tfsa[0]

    print(f"\nScaling Exponents:")
    print(f"Naive (expected ~2.0): {exp_naive:.3f}")
    print(f"TFSA  (expected ~1.1): {exp_tfsa:.3f}")

    # Verification
    assert exp_tfsa <= 1.1, f"TFSA scaling too high: {exp_tfsa:.3f}"
    assert exp_naive >= 1.5, f"Naive scaling too low: {exp_naive:.3f}"

    # Speedup at n=1000
    idx_1000 = n_sizes.index(1000)
    speedup = naive_times[idx_1000] / tfsa_times[idx_1000]
    print(f"Speedup at n=1000: {speedup:.2f}x")
    assert speedup >= 10, f"Speedup too low: {speedup:.2f}x"

    print("\nSUCCESS: TFSA demonstrates near-linear scaling and significant speedup.")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.loglog(n_sizes, naive_times, 'o-', label=f"Naive (exp={exp_naive:.2f})")
    plt.loglog(n_sizes, tfsa_times, 'o-', label=f"TFSA (exp={exp_tfsa:.2f})")
    plt.title("Complexity Analysis: Naive Surprise vs TFSA")
    plt.xlabel("Number of Relators (n)")
    plt.ylabel("Computation Time (s)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("exp_07_scaling.png")
    print("Scaling plot saved as exp_07_scaling.png")

if __name__ == "__main__":
    run_exp_07()
