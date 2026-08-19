# Independent Design Challenge: Selective Executable Probing

## Purpose

This is a design review before the final simulator run. It asks whether the experiment is fair, whether its failure conditions cover
the main technical risks, and whether hidden simulator information could reach an ordinary policy.

No development scores, effect sizes, policy rankings, or final outcomes are included. The review is not evidence that the simulator
represents real learners.

The review should take about 15 minutes. No code execution or statistical calculation is required.

## Research Question

> Across predeclared matched and misspecified simulator environments, can a reliability-aware Bayesian policy use a fixed 50% probe
> budget to reduce later-task prediction error relative to random, uncertainty-only, and ordinary value-of-information selection?

A **probe** is an optional request for executable programming evidence. In this study, its pass, fail, or missing result is simulated.
The policy decides which 20 of 40 possible probes to request. It does not choose a teaching response.

## What One Episode Contains

One simulated episode contains 40 cases, arranged as 10 families of four related cases. Each case has:

- a prior probability of later success;
- one of four probe classes;
- a hidden later-success outcome;
- an optional probe outcome: pass, fail, or missing.

The hidden outcome is used only for scoring after a policy has selected its probes and committed its predictions. Failures may be
linked within a family, so the episode, rather than each case, is the independent unit in the analysis.

## Policy Under Review

Calibration data estimate the sensitivity and specificity of each probe class. These estimates remain uncertain. For each case, the
candidate policy estimates how much a probe might reduce classification error, then uses a cautious lower estimate rather than the
average estimate. A single probe can change posterior log-odds by no more than `2.0`.

The candidate must select exactly 20 probes, even when some estimated benefits are negative. Ties are resolved by case identifier.
Calibration is frozen before evaluation and is never updated from evaluation outcomes.

## Comparisons

All matched-budget policies see the same 40 cases and select exactly 20 probes:

| Policy | Selection rule |
|---|---|
| Seeded random | Selects a reproducible random set |
| Uncertainty only | Selects cases whose prior is closest to 0.5 |
| Ordinary value of information | Uses average calibrated probe reliability |
| Reliability aware | Uses a cautious lower estimate of value while reliability is uncertain |

Never probing and always probing provide burden and performance endpoints. An oracle that knows the true reliability is reported
only as unattainable headroom, not as a fair competitor.

The primary measurement is final classification error at the 50% budget. Policies are paired within each episode, so they face the
same cases, hidden outcomes, and available probe results.

## Simulator Environments

The policy is calibrated for the matched environment. It is not told which environment generated an evaluation episode.

| Environment | What changes | Reason for including it |
|---|---|---|
| Matched | Probe reliability follows the calibration model | Checks that the implementation behaves under its own assumptions |
| Lower reliability | Every probe class is moved halfway from its calibrated reliability towards chance | Tests broad overconfidence in the evidence channel |
| Asymmetric errors | Sensitivity is reduced much more than specificity | Tests a channel that misses genuine success more often than expected |
| Irrelevant evidence | Probe outcomes are independent of later success | Tests evidence that looks valid but carries no useful signal |
| Inverted evidence | Every sampled pass or fail is reversed | Deliberately severe negative control for misleading evidence |
| Difficulty-linked missingness | Missing results become more likely when prior success is near 0.5, up to 60% | Tests whether evidence disappears on the cases that appear most uncertain |
| Correlated family failures | All four probe results in a family are reversed together with probability 0.30 | Tests failures that are linked rather than independent |

The six non-matched environments are stress tests chosen by the researcher. Their values were not estimated from learners and must
not be described as classroom prevalence.

## Information Boundary

An ordinary policy may receive only:

- case identifier;
- prior success probability;
- probe class;
- calibration-derived reliability distributions;
- remaining probe budget.

It must not receive:

- family identifier;
- evaluation environment identifier;
- true evaluation reliability or missingness settings;
- hidden later-success outcome;
- an unrequested probe result;
- any criterion or future outcome.

The oracle is the sole exception. It receives true reliability for reference, but never receives realised hidden answers before
selection.

## Evaluation Rules

- Development and evaluation use separate episode ranges.
- The final run contains 2,000 episodes in each of seven environments.
- The six held-out environments receive equal weight in the primary summary.
- The matched environment is reported separately and cannot establish robustness by itself.
- Candidate comparisons use paired episode differences and predeclared simultaneous confidence intervals.
- No episode is excluded after generation.
- A software failure invalidates the run; a scientific failure is retained.
- Evaluation outcomes cannot alter policies, parameters, thresholds, exclusions, or environment definitions.

## Claim Boundary

The strongest permitted claim is about prediction error and probe use inside these declared simulator environments. Even a positive
result would not show improved tutoring, human learning, learner mastery, reduced cognitive offloading, or financial savings in a
deployed system.

## Material Supplied

1. `review-packet.md`, this plain-language design summary.
2. `environment-provenance.csv`, showing why each stress environment was included.
3. `review-response-template.md`, the response form.

Please complete the response form without requesting development results. A concern needs only a short explanation. A material
change means the frozen design must be versioned and rehearsed again before the final run.

## Frozen Design References

- Environment specification hash: `4f72b0944ea61598408910498fb5d017db9ee1f7ac09cc1a2e01edcf42a0191b`
- Analysis specification hash: `6dcfcb630f5f33d14b4f8d8eb64019505a803d6350b9b8d2ffed82afc3350919`

These identify the reviewed design. They do not reveal development or evaluation results.
