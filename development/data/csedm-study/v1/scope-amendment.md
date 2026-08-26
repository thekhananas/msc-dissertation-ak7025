# M7C Scope Amendment

Date: 9 September 2026

## Research question

On the official learner-separated CSEDM folds, does ranking out-of-fold predictions by uncertainty capture more first-attempt prediction errors within a fixed 50% review budget than the exact expectation under uniform random selection?

## Why this study was added

The earlier studies used an authored benchmark and controlled simulations. They established whether the software worked and exposed ways in which executable evidence could mislead a tracker, but they could not show whether uncertainty is useful on records from real learners.

M7C adds one bounded check using historical novice Python data. It tests whether uncertain predictions are a useful way to identify cases that deserve review.

## Boundaries

This study predicts first-attempt success from activity recorded before each target problem. It does not request executable evidence, change a tutoring action, or measure learning caused by a tutor. It is therefore a companion to M7B, not a replication of it.

The result may support a claim about prediction-error triage on this dataset. It cannot support claims about probe benefit, reduced cost, improved teaching, cognitive offloading, or human learning.

The Python CSEDM Data Challenge archive is the only dataset in scope. The separate Java CodeWorkout release under `development/data/All/` is excluded.

## Time and stopping rule

M7C has a hard limit of 12 elapsed working hours, including defect correction and documentation. It uses no external model, sandbox, GPU, neural model, or new interface.

If the source inventory, analysis plan, leakage-safe features, all official folds, analysis, and reproducible public outputs are not complete at hour 12, the study stops. Partial or selectively favourable results remain outside the abstract and conclusion. M8 resumes immediately.

## Delivery record

- Scope recorded before model fitting or inspection of out-of-fold results.
- Implementation may begin while supervisor review is pending.
- Without supervisor acceptance before the dissertation results freeze, M7C will be labelled exploratory.
- Current supervisor status: pending.
