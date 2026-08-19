# M7B Independent Design Challenge: Disposition

**Review date:** 6 September 2026  
**Reviewer identifier:** `OT`  
**Reviewer description:** Self-described ML experience  
**Overall response:** Proceed unchanged  
**Material design change required:** No

## Review Integrity

The reviewer reported completing the review independently and without seeing development outcomes or policy rankings. The raw
response is preserved without correction at `review-submissions/reviewer-completed.md`, SHA-256
`cb0675bebd9ad9bcdf75573f8c591e1588a28c2a797d6776c6a4b2dbf6985d95`.

The packet and response were not committed in separate stages. The repository will therefore preserve their content, but its commit
history alone cannot prove that the packet commit predates the response. The canonical evaluation has not been run, and no result was
available to the reviewer.

## Response And Action

| Question | Reviewer response | Disposition |
|---|---|---|
| Distinct environments | Yes | No change. All seven settings alter a different declared part of the evidence process. |
| Plausible technical weaknesses | Yes, with a note that sampling matters | No change. The final design retains 2,000 independent episodes per environment and reports the hand-specified sampling model as a limit. |
| Missing evidence failure | Unsure; no missing condition was named | Do not add a condition after development on an unspecified concern. State that the stress set is deliberately broad but not exhaustive. |
| Hidden-environment inference | Unsure | Direct access is blocked by the policy contract and safe projection. The claim is limited to fixed, audited policies, not hostile code capable of reading configuration files or reverse-engineering seeds. |
| Outcome leakage before selection | No | No concern identified. Hidden outcomes and unrequested evidence remain in the privileged record until scoring. |
| Fair fixed-budget comparison | Yes | No change. Every matched-budget policy selects 20 probes from the same 40 cases. |
| Episode as independent unit | Unsure | Retain the episode as the unit because dependence is permitted within its ten families, while episodes use independent random draws. Report this assumption and do not analyse 40 cases as independent samples. |
| Oracle boundary | Yes | No change. The oracle remains unattainable headroom, never a fair comparator. |
| Human and deployment claim limits | Yes | No change. Simulator results cannot support claims about learning, tutoring efficacy, cognitive offloading, or deployed savings. |
| Material change before final run | No | Proceed to the pre-run seal once the technical gates pass. |

## Technical Evidence For The Information Boundary

Existing focused checks confirm that policy-facing records contain no environment identifier, family identifier, hidden success,
effective sensitivity or specificity, or probe outcome. The frozen ordinary policies accept only an `AcquisitionRequest`; they are
stateless and do not load an environment configuration. The 12 relevant contract, plan, and simulator checks passed on 6 September
2026.

This protects the declared implementation, not arbitrary untrusted policy code. The pre-run package must therefore hash the exact
policy implementation and must not permit dynamic policy loading during canonical evaluation.

## Decision

Close `M7B.2.4` with no experiment change. Carry three limits into the final report: the simulator settings are hand-specified, the
failure set is not exhaustive, and information isolation applies to the frozen audited policies rather than adversarial code.
