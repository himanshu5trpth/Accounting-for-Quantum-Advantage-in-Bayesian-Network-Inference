# Amendment 002 — post-primary robustness confirmation

The completed primary study found frequent cap noncompletion at 16 shots per IAE round, even for the interval policy. A deterministic reproduction traced the exceptions to the prespecified 200-round cap, rather than a table-join or interval-arithmetic error. The primary outcomes will not be replaced or retried.

Before this follow-up is executed, fix one additional setting: the same interval policy with 100 shots per round, the historical default shot count. Keep the 200-round and 1e10 Grover-power caps, total alpha 0.05, eight refinements, all 92 selected queries and both tolerance regimes. Evaluate 256 independent repetitions per query-setting, base seed 910000000, disjoint from primary tuning and evaluation. There is no new parameter search, query filtering or stopping on significance.

This is a post-primary implementation robustness study, not an originally registered primary hypothesis or a new independent patient cohort. Report its outcomes alongside the complete failed 16-shot results. Apply Holm over the 184 follow-up per-query coverage tests. Any transfer or population recalculation based on this alternative remains a sensitivity analysis until validated on a representative independent query population.

The exact classical reader also normalizes rounded public BIF probability rows before sampling. Independent saved-posterior and exact-variance checks remain mandatory and passed on every eligible query. This is a numerical input representation correction, not a relaxation based on coverage outcomes.
