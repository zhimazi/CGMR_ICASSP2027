"""Tiny model-free demonstration of the two central CGMR operations."""

from cgmr import classify_state, project_language_cohort


# One row is ranking-accessible: the adapted 1-best regresses from 2 to 4 edits,
# while a 2-edit alternative is still present in the beam.
print("state:", classify_state(2, 4, 2))

# Two state-A rows from one language. Each tuple is
# (adapted-model candidate log scores, reference edit counts).
rows = [
    ([0.0, -0.4, -1.0], [4, 2, 0]),
    ([0.0, -0.2, -0.8], [3, 1, 0]),
]
projection = project_language_cohort(rows, comparator_budget=2.0)

print("lambda:", round(projection.lambda_value, 6))
print("risk before:", round(projection.risk_before, 6))
print("risk after:", round(projection.risk_after, 6))
for index, q in enumerate(projection.posteriors, start=1):
    print(f"q[{index}] =", [round(float(x), 6) for x in q])
