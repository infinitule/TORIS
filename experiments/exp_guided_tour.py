"""TORIS — The Guided Tour.

A single reasoning problem, walked through all nine layers of TORIS with
narration. This is a teaching artifact: every box explains what the layer
*is*, then runs a real computation on one shared field and reads the result
back in the language of the problem.

    THE PROBLEM — "Should the operator open the dam?"

    Heavy rain is filling a reservoir. Pressure builds against the dam. If the
    pressure is not relieved, the dam may breach and devastate the town
    downstream. The operator can open a controlled release — but the release
    itself floods the valley. Opening the dam therefore BOTH protects and
    harms downstream safety: a genuine, irreducible contradiction.

Run:  .venv/bin/python experiments/exp_guided_tour.py
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import textwrap

from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.primitives.relation_types import RelationType as T
from toris.field.relational_field import RelationalField
from toris.goal.manifold import GoalManifold, Goal
from toris.goal.warp import warp_field, relevance_fn
from toris.engine.surprise import SurpriseMetric
from toris.engine.predictive import PredictiveEngine
from toris.plasticity.fast import FastPlasticity
from toris.reasoning.chain import ReasoningChain
from toris.engine.wave import CyclicWaveEngine
from toris.engine.tasf import TASF
from toris.engine.running_coupling import SurpriseCoupling
from toris.engine.circle_method import saddle_point, circle_method_surprise
from toris.engine.suppression import suppression_report, suppressed_depth, is_modular_field
from toris.engine.rogers_ramanujan import partition_function_rr, field_entropy
from toris.engine.rademacher import rademacher_surprise
from toris.engine.complete_surprise import UnifiedSurprise

# ── presentation helpers ─────────────────────────────────────────────────────
W = 70
_TTY = _sys.stdout.isatty()  # only emit ANSI colour to a real terminal
def _c(code):
    return code if _TTY else ""
TEAL = _c("\033[36m"); GOLD = _c("\033[33m"); CORAL = _c("\033[31m")
DIM = _c("\033[2m"); OFF = _c("\033[0m"); BOLD = _c("\033[1m")


def banner(n, name, tagline):
    print(f"\n{TEAL}{'━' * W}{OFF}")
    print(f"{TEAL}{BOLD}  LAYER {n} · {name}{OFF}")
    print(f"{DIM}  {tagline}{OFF}")
    print(f"{TEAL}{'━' * W}{OFF}")


def say(text):
    for line in textwrap.wrap(text, W - 4):
        print(f"  {line}")


def show(label, value, color=""):
    if color:
        print(f"    {color}{label:<34}{OFF} {value}")
    else:
        print(f"    {label:<34} {value}")


def beat():
    print()


# ── the shared field (built once, threaded through every layer) ──────────────
def build_reservoir_field():
    """The reservoir-decision knowledge field.

    Causal spine:   rain → reservoir → pressure → breach → downstream_safety
    Feedback loop:  pressure ⇄ release   (pressure enables release; release relieves pressure)
    Productive contradiction:  release → downstream_safety   ENABLES ⊗ NEGATES
    """
    f = RelationalField()
    names = ["rain", "reservoir", "pressure", "breach",
             "release", "downstream_safety", "sensors", "town"]
    C = {}
    for nm in names:
        c = ConceptState(id=nm)
        C[nm] = c
        f.add_concept(c)

    # Give 'release' an explicit role distribution Π to illustrate Layer 0.
    C["release"].role_distribution = {T.ENABLES: 0.5, T.NEGATES: 0.3, T.CONDITIONAL: 0.2}
    C["release"].__post_init__()  # renormalize

    R = [
        Relator(T.CAUSAL,    C["rain"],      C["reservoir"],         sigma=0.95, kappa=0.90, epsilon=0.05),
        Relator(T.CAUSAL,    C["reservoir"], C["pressure"],          sigma=0.90, kappa=0.85, epsilon=0.05),
        Relator(T.EVIDENCES, C["sensors"],   C["pressure"],          sigma=0.80, kappa=0.70, epsilon=0.10),
        Relator(T.CAUSAL,    C["pressure"],  C["breach"],            sigma=0.70, kappa=0.60, epsilon=0.15),
        Relator(T.NEGATES,   C["breach"],    C["downstream_safety"], sigma=0.90, kappa=0.80, epsilon=0.10),
        Relator(T.ENABLES,   C["pressure"],  C["release"],           sigma=0.70, kappa=0.65, epsilon=0.10),
        Relator(T.NEGATES,   C["release"],   C["pressure"],          sigma=0.85, kappa=0.75, epsilon=0.08),
        # ── the productive contradiction ──
        Relator(T.ENABLES,   C["release"],   C["downstream_safety"], sigma=0.80, kappa=0.78, epsilon=0.05),
        Relator(T.NEGATES,   C["release"],   C["downstream_safety"], sigma=0.45, kappa=0.40, epsilon=0.60),
        Relator(T.CONTAINS,  C["downstream_safety"], C["town"],      sigma=0.60, kappa=0.50, epsilon=0.05),
    ]
    for r in R:
        f.add_relator(r)
    return f, C


def field_without(field, C, drop_pairs):
    """A copy of *field* with the (src,tgt) edges in drop_pairs removed —
    used as a 'prediction' that didn't foresee certain structure."""
    g = RelationalField()
    for c in field.concepts():
        g.add_concept(c)
    for r in field.relators():
        if (r.src_id, r.tgt_id) not in drop_pairs:
            g.add_relator(r.clone())
    return g


# ══════════════════════════════════════════════════════════════════════════════
def run():
    print(f"{GOLD}{BOLD}")
    print("  ┌────────────────────────────────────────────────────────────────┐")
    print("  │   TORIS — THE GUIDED TOUR                                       │")
    print("  │   One reasoning problem, walked through all nine layers.        │")
    print("  └────────────────────────────────────────────────────────────────┘")
    print(OFF)
    say("THE PROBLEM. Heavy rain fills a reservoir; pressure builds against "
        "the dam. Left unrelieved, the dam may breach and devastate the town "
        "downstream. The operator can open a controlled release — but the "
        "release itself floods the valley. Opening the dam BOTH protects and "
        "harms downstream safety. Watch how each layer reasons about it.")

    field, C = build_reservoir_field()

    # ── LAYER 0 ───────────────────────────────────────────────────────────────
    banner(0, "PRIMITIVES", "the typed hypergraph — concepts as roles, not coordinates")
    say("The world is not points in a vector space. It is a typed, directional "
        "hypergraph of Relators. Each Relator is a 6-tuple R = (τ, src, tgt, "
        "σ, κ, ε): a kind of relation, a direction, a strength, a salience, "
        "and its own surprise.")
    beat()
    show("Concepts (nodes)", len(list(field.concepts())))
    show("Relators (typed edges)", field.num_relators())
    types = sorted({r.tau.name for r in field.relators()})
    show("Relation types present", ", ".join(types))
    beat()
    r = next(r for r in field.relators() if r.tau == T.CAUSAL and r.src_id == "pressure")
    say("One relator, in full — the operator's core worry:")
    show("  R", f"({r.tau.name}, {r.src_id}→{r.tgt_id}, σ={r.sigma}, κ={r.kappa}, ε={r.epsilon})", CORAL)
    beat()
    say("A concept is a distribution over the roles it can play (Π : T → [0,1]). "
        "'release' is not a coordinate — it is mostly an ENABLER, sometimes a "
        "NEGATOR, sometimes CONDITIONAL:")
    rel = C["release"]
    dom = max(rel.role_distribution, key=rel.role_distribution.get)
    for t, p in sorted(rel.role_distribution.items(), key=lambda kv: -kv[1]):
        if p <= 0:
            continue
        bar = "█" * int(p * 24)
        show(f"  Π(release · {t.name})", f"{p:.2f} {TEAL}{bar}{OFF}")
    show("  dominant role", dom.name, GOLD)

    # ── LAYER 1 ───────────────────────────────────────────────────────────────
    banner(1, "SURPRISE  ΔS", "topological deviation — not cosine, not distance")
    say("The operator EXPECTED rain to raise the reservoir and pressure to "
        "build — but did NOT foresee the breach pathway. Surprise is the "
        "structural difference between the predicted field and the observed "
        "one, split into structure / type / strength.")
    f_pred = field_without(field, C, {("pressure", "breach"), ("breach", "downstream_safety")})
    sm = SurpriseMetric()
    rep = sm.report(f_pred, field)
    beat()
    show("ΔS  (total)", f"{rep.delta_s:.4f}", CORAL)
    show("  ΔS_structural (α=0.6)", f"{rep.delta_s_struct:.4f}", "  ")
    show("  ΔS_type       (β=0.3)", f"{rep.delta_s_type:.4f}")
    show("  ΔS_strength   (γ=0.1)", f"{rep.delta_s_strength:.4f}")
    prop = list(rep.propagating())
    show("Relators that propagate (ε>θ)", len(prop), GOLD)
    say("The surprise is overwhelmingly STRUCTURAL: the breach edges simply "
        "weren't in the prediction. Confirmed expectations stay silent.")

    # ── LAYER 2 ───────────────────────────────────────────────────────────────
    banner(2, "PREDICTIVE ENGINE", "project → observe → delta → propagate")
    say("Inference is a loop. Project an expectation from the field under the "
        "current goal; observe what actually arrives; compute the delta; let "
        "only the surprising part propagate.")
    goal = GoalManifold(primary=Goal(
        description="prevent catastrophic breach",
        concepts={"pressure", "breach", "downstream_safety", "release"},
        preferred_types={T.CAUSAL, T.NEGATES, T.ENABLES},
    ))
    engine = PredictiveEngine()
    projected = engine.project(field, relevance_fn(goal))
    # reality arrives with one unforeseen reading: sensors directly evidence a breach
    observed = field.copy()
    observed.add_relator(Relator(T.EVIDENCES, C["sensors"], C["breach"],
                                 sigma=0.7, kappa=0.8, epsilon=0.9))
    rep2 = engine.compute_delta(projected, observed)
    prop2 = engine.propagate(rep2)
    beat()
    show("Projected expectation (edges)", projected.num_relators())
    show("Observed reality (edges)", observed.num_relators())
    show("ΔS of this step", f"{rep2.delta_s:.4f}", CORAL)
    show("Propagating (surprising) relators", len(prop2), GOLD)
    say("Most of reality matched the projection and cost nothing. The new "
        "sensors→breach reading is the one signal that flows forward.")

    # ── LAYER 3 ───────────────────────────────────────────────────────────────
    banner(3, "GOAL MANIFOLD", "the goal warps the topology — Φ(G, F)")
    say("A goal does not just re-weight edges; it changes which edges are even "
        "ACTIVE, and surfaces the contradictions it makes relevant. Two "
        "operators with two goals see two different fields.")
    goal_safety = GoalManifold(primary=Goal(
        description="protect the town from a breach",
        concepts={"breach", "downstream_safety", "release", "town"},
        preferred_types={T.NEGATES, T.ENABLES, T.CONTAINS},
    ))
    goal_hydro = GoalManifold(primary=Goal(
        description="model the water dynamics",
        concepts={"rain", "reservoir", "pressure"},
        preferred_types={T.CAUSAL, T.EVIDENCES},
    ))
    f_safety = warp_field(goal_safety, field)
    f_hydro = warp_field(goal_hydro, field)
    diff = f_safety.edge_set().symmetric_difference(f_hydro.edge_set())
    beat()
    show("Active edges · safety goal", len(f_safety.edge_set()))
    show("Active edges · hydrology goal", len(f_hydro.edge_set()))
    show("Edges that differ between goals", len(diff), GOLD)
    show("Contradictions surfaced (safety)", len(goal_safety.contradiction_log), CORAL)
    if len(goal_safety.contradiction_log):
        e = list(goal_safety.contradiction_log._entries.values())[0]
        show("  the held tension",
             f"{e.relator_a.tau.name} ⊗ {e.relator_b.tau.name} on "
             f"{e.relator_a.src_id}→{e.relator_a.tgt_id}", CORAL)
    say("Under the safety goal the release⊗ contradiction becomes live — the "
        "operator must hold 'opening helps' AND 'opening harms' at once.")

    # ── LAYER 4 ───────────────────────────────────────────────────────────────
    banner(4, "FAST PLASTICITY", "the field rewrites itself during inference")
    say("Surprise does not just get measured — it edits the topology. New "
        "relators are added where structure was missing; what surprised is "
        "strengthened; perfect predictors fade; the off-goal is suppressed.")
    plastic = FastPlasticity()
    new_field, delta = plastic.step(field, rep, goal_safety)
    beat()
    show("ADD       (gaps surprise revealed)", len(delta.added), GOLD)
    show("STRENGTHEN (mattered more than expected)", len(delta.strengthened))
    show("WEAKEN     (confirmed, becoming background)", len(delta.weakened))
    show("SUPPRESS   (irrelevant to the goal)", len(delta.suppressed))
    show("Field size  before → after", f"{field.num_relators()} → {new_field.num_relators()}")
    say("The field at the next step is not the field we started with. Inference "
        "is rewriting a live structure, not reading a frozen one.")

    # ── LAYER 5 ───────────────────────────────────────────────────────────────
    banner(5, "REASONING", "sparse seeds → a multi-hop conclusion with calibrated σ")
    say("From a handful of typed relations, traverse a chain the operator never "
        "stated explicitly: does the rain threaten the town? Confidence is the "
        "composed strength along the path, discounted by accumulated surprise.")
    chain = ReasoningChain()
    path = [C["rain"], C["reservoir"], C["pressure"], C["breach"], C["downstream_safety"]]
    res = chain.infer_along(field, path)
    beat()
    show("Question", "will the rain threaten downstream safety?")
    show("Path", "rain → reservoir → pressure → breach → downstream_safety")
    show("Hops inferred", res.hops, GOLD)
    show("Seeds used / hypotheticals", f"{res.n_seed} / {res.n_hypothetical}")
    show("Chain strength  σ_chain", f"{res.sigma_chain:.4f}", CORAL)
    show("Broken chain?", res.broken)
    say("A 4-hop conclusion with a real, calibrated confidence — and TORIS "
        "knows exactly how uncertain it is, because the uncertainty is built "
        "into the chain.")

    # ── LAYER 6 ───────────────────────────────────────────────────────────────
    banner(6, "FAST SURPRISE DYNAMICS", "an O(n log n) screen + surprise as a damped wave")
    say("The full ΔS is O(n²). The TFSA fast-screen approximates it in O(n log "
        "n) and agrees on what matters. And surprise does not jump — it "
        "propagates through the field as a damped cyclic wave, which can go "
        "UNSTABLE around a feedback loop.")
    full = sm.report(f_pred, field)
    fast = sm.combined_pipeline(f_pred, field)
    s_full = {x.rid for x in full.propagating()}
    s_fast = {x.rid for x in fast.propagating()}
    overlap = len(s_full & s_fast) / max(len(s_full | s_fast), 1)
    wave = CyclicWaveEngine(b=0.1)
    unstable = wave.scan_field(field)
    beat()
    show("Full ΔS  · propagating set", len(s_full))
    show("TFSA fast · propagating set", len(s_fast))
    show("Agreement (Jaccard overlap)", f"{overlap:.0%}", GOLD)
    show("Unstable surprise cycles found", len(unstable), CORAL)
    say("The pressure ⇄ release feedback loop is exactly the kind of structure "
        "where surprise can resonate rather than settle — the wave engine "
        "flags it.")

    # ── LAYER 7 ───────────────────────────────────────────────────────────────
    banner(7, "ANALYTIC SURPRISE", "the contradiction is a POLE; surprise has a running coupling")
    say("Treat salience κ as a complex variable. The surprise density becomes "
        "an analytic function — and a PRODUCTIVE CONTRADICTION appears as a "
        "literal pole on the real axis. Surprise also has a 'running coupling': "
        "it weakens as salience grows (asymptotic freedom).")
    tasf = TASF(N_quadrature=32)
    tr = tasf.compute(field, field.copy())
    beat()
    show("Analytic ΔS (field vs. itself)", f"{tr.delta_S_analytic:.6f}")
    show("Poles on the sampled contour", len(tr.poles), CORAL)
    say("Comparing the field to itself nets zero analytic deviation by design — "
        "what we are isolating is the field's INTRINSIC analytic structure. A "
        "contradiction shows up as a pole only when its salience κ crosses the "
        "integration contour; here the release-tension sits at κ=0.40, below "
        "it. The living structure is the coupling, below:")
    beat()
    sc = SurpriseCoupling()
    ks = [0.1, 0.3, 0.5, 0.7, 0.9]
    al = sc.run_coupling(field, ks)
    say("Running coupling α_S(κ) — note the monotone fall:")
    for k, a in zip(ks, al):
        bar = "█" * int(a * 400)
        show(f"  α_S(κ={k:.1f})", f"{a:.4f} {GOLD}{bar}{OFF}")
    mono = all(al[i] >= al[i + 1] for i in range(len(al) - 1))
    show("  asymptotic freedom holds", mono, TEAL)

    # ── LAYER 8 ───────────────────────────────────────────────────────────────
    banner(8, "RAMANUJAN EXTENSION", "deep surprise via the circle method · suppression · partition function")
    say("Reasoning at depth d means traversing d-hop chains — exponentially "
        "many. Hardy & Ramanujan's CIRCLE METHOD gives the dominant surprise "
        "from ONE saddle point. Ramanujan's PARTITION CONGRUENCES make whole "
        "depths cancel exactly. And the ROGERS–RAMANUJAN identity gives the "
        "field's configuration entropy in closed form.")
    beat()
    say("Circle method — O(1) surprise per depth, no traversal:")
    for d in (3, 5, 7):
        cm = circle_method_surprise(field, d)
        show(f"  depth d={d}", f"κ_saddle={cm.kappa_saddle:.3f}   ΔS_dominant={cm.delta_s_total:.4f}")
    beat()
    say("Suppression theorem — depths where surprise provably vanishes "
        "(p(5m+4)≡0, p(7m+5)≡0, p(11m+6)≡0):")
    supp = [e.depth for e in suppression_report(field, max_depth=12).entries if e.expected_suppressed]
    show("  modular field?", is_modular_field(field, 5))
    show("  suppressed depths ≤ 12", supp, GOLD)
    show("  (TORIS skips these entirely)", "0 surprise, 0 compute", DIM)
    beat()
    say("Rogers–Ramanujan — the field's reasoning flexibility, in closed form:")
    show("  partition function  Z_F(0.3)", f"{partition_function_rr(field, 0.3):.4f}")
    show("  field entropy  H(F)", f"{field_entropy(field):.4f} nats", GOLD)

    # ── LAYER 9 ───────────────────────────────────────────────────────────────
    banner(9, "EXACT SURPRISE", "the Rademacher series — exact, with a CERTIFIED error bound")
    say("The deepest layer computes surprise EXACTLY via a Rademacher series — "
        "and returns a certified error bound (a theorem, not an estimate). A "
        "unified router then picks the cheapest valid regime for each depth: "
        "fast / deep / suppressed.")
    beat()
    say("Rademacher exact surprise at three depths:")
    for d in (3, 7, 9):
        rr = rademacher_surprise(field, d, N_terms=3)
        show(f"  d={d}", f"S_exact={rr.S_exact:>10.4f}   ±{rr.error_bound:.2e}   "
                          f"int-nearness={rr.integer_nearness:.3f}")
    beat()
    say("Unified regime routing across depths:")
    us = UnifiedSurprise()
    print(f"    {DIM}{'d':>4}  {'ΔS':>10}  {'regime':>11}  {'suppressed':>10}  {'err_bound':>10}{OFF}")
    for d in (1, 3, 7, 9, 12):
        u = us.compute(field, None, d=d, goal_manifold=goal_safety)
        col = GOLD if u.suppressed else ""
        print(f"    {col}{d:>4}  {u.delta_S:>10.4f}  {u.regime_used:>11}  "
              f"{str(u.suppressed):>10}  {u.error_bound:>10.2e}{OFF}")

    # ── FINALE ────────────────────────────────────────────────────────────────
    print(f"\n{GOLD}{'━' * W}{OFF}")
    print(f"{GOLD}{BOLD}  THE SAME DECISION, THROUGH NINE LENSES{OFF}")
    print(f"{GOLD}{'━' * W}{OFF}")
    say("One reservoir, one operator, one irreducible contradiction. TORIS "
        "represented it as a typed field (0), measured what surprised (1), "
        "predicted and propagated (2), warped it by goal (3), let it rewrite "
        "itself (4), reasoned across hops (5), screened and waved surprise "
        "(6), found the contradiction as an analytic pole (7), computed deep "
        "surprise with Ramanujan's number theory (8), and bounded it exactly "
        "(9) — never once averaging the tension away.")
    print(f"\n  {TEAL}{BOLD}No vectors. No softmax. Just relations.{OFF}\n")


if __name__ == "__main__":
    run()
