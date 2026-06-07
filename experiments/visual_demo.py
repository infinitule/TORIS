import time
from toris.primitives.concept_state import ConceptState
from toris.primitives.relator import Relator
from toris.primitives.relation_types import RelationType
from toris.field.relational_field import RelationalField
from toris.engine.surprise import SurpriseMetric
from toris.visualizer import FieldVisualizer

def run_visual_demo():
    print("Generating TORIS Dynamic Visual Demo...")

    # 1. Base Field
    field = RelationalField()
    c1 = ConceptState(id="A")
    c2 = ConceptState(id="B")
    c3 = ConceptState(id="C")
    c4 = ConceptState(id="D")

    # Add some base structure
    field.add_relator(Relator(RelationType.CAUSAL, c1, c2, sigma=0.8, kappa=0.5))
    field.add_relator(Relator(RelationType.CAUSAL, c2, c3, sigma=0.8, kappa=0.5))
    field.add_relator(Relator(RelationType.CONTAINS, c1, c4, sigma=0.9, kappa=0.5))

    viz = FieldVisualizer(field)
    viz.export_html("toris_step_0_base.html", "TORIS: Step 0 - Base Field")
    print("Saved: toris_step_0_base.html")

    # 2. Goal Warp
    # Simulate Goal shift that amplifies C2->C3 and suppresses C1->C4
    print("Applying Goal Warp...")
    for r in field.relators():
        if r.tgt_id == "C": r.kappa = 0.9
        if r.tgt_id == "D": r.kappa = 0.1

    viz.export_html("toris_step_1_warped.html", "TORIS: Step 1 - Goal Warped")
    print("Saved: toris_step_1_warped.html")

    # 3. Surprise Event
    print("Injecting Surprise...")
    # New unexpected relation appearing
    r_surprise = Relator(RelationType.CONTRADICTS, c3, c1, sigma=0.7, kappa=0.9, epsilon=0.8)
    field.add_relator(r_surprise)

    viz.export_html("toris_step_2_surprised.html", "TORIS: Step 2 - Surprise Event")
    print("Saved: toris_step_2_surprised.html")

    # 4. Plasticity
    print("Applying Structural Plasticity...")
    # C3 now causes a new concept E
    c5 = ConceptState(id="E")
    field.add_relator(Relator(RelationType.CAUSAL, c3, c5, sigma=0.5, kappa=0.8))

    viz.export_html("toris_step_3_plastic.html", "TORIS: Step 3 - Structural Plasticity")
    print("Saved: toris_step_3_plastic.html")

    print("\nDemo Complete. Please open the .html files in your browser to see the evolution.")

if __name__ == "__main__":
    run_visual_demo()
