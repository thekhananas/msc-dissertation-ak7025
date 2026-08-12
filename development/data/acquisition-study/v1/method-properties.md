# Reliability-Aware Probe Selection: Method and Limits

## Scope

This note states the mathematics used by the M7B acquisition policy. The method selects executable probes inside a simulator. It
does not establish that the simulator represents a learner, that probing improves learning, or that fewer simulated probes produce
real deployment savings.

## Probe Model

For one candidate, let \(K\in\{0,1\}\) denote whether the later task will be solved, and let
\(p=P(K=1)\). A binary probe result \(Z\in\{0,1\}\) has sensitivity and specificity

\[
s=P(Z=1\mid K=1),
\qquad
c=P(Z=0\mid K=0).
\]

The predicted probability of a passing probe is

\[
q=P(Z=1)=ps+(1-p)(1-c).
\]

Calibration data produce separate Beta posteriors for \(s\) and \(c\). Evaluation outcomes do not update them.

## Bounded Update

For a passing probe, the log-likelihood change is \(\log(s/(1-c))\). For a failing probe, it is
\(\log((1-s)/c)\). The implemented update is

\[
T_\kappa(p,z;s,c)
=
\operatorname{logistic}\!\left(
\operatorname{logit}(p)
+
\operatorname{clip}(\Delta_z,-\kappa,\kappa)
\right).
\]

It follows directly that

\[
\left|
\operatorname{logit}(T_\kappa(p,z;s,c))-\operatorname{logit}(p)
\right|
\leq \kappa.
\]

This limits how far one probe can move the belief. It does not guarantee that the movement is correct.

## Acquisition Score

Under equal costs for the two classification errors, immediate risk is

\[
R(p)=\min(p,1-p),
\qquad 0\leq R(p)\leq \tfrac12.
\]

For fixed \((s,c)\), the expected reduction in this risk is

\[
g(p,s,c)
=R(p)
-qR\!\left(T_\kappa(p,1;s,c)\right)
-(1-q)R\!\left(T_\kappa(p,0;s,c)\right).
\]

The plug-in baseline evaluates \(g\) at the posterior means of \(s\) and \(c\). The proposed policy draws
\((s,c)\) from their calibration posteriors and uses the lower \(0.10\) quantile of \(g\). It therefore favours probes whose
estimated benefit survives less favourable, but still plausible, reliability values.

The fixed-budget policy must select the declared number of probes, even if a score is negative. A negative score is retained and
reported; it is not silently clipped to zero.

## Hand-Worked Check

Take \(p=0.4\), \(s=0.8\), \(c=0.7\), and \(\kappa=2\). Then \(q=0.5\). Neither likelihood ratio is clipped, and

\[
T_\kappa(p,1;s,c)=0.64,
\qquad
T_\kappa(p,0;s,c)=0.16.
\]

The prior risk is \(0.4\), while expected posterior risk is

\[
0.5(0.36)+0.5(0.16)=0.26.
\]

The plug-in expected risk reduction is therefore \(0.14\). This value is fixed in an executable check.

## Limited Guarantees

1. The change in posterior log-odds from one probe cannot exceed \(\kappa\) in magnitude.
2. Classification risk remains between zero and one half for every valid probability.
3. A frozen seed, draw count, quantile rule, and case-ID tie break make the ranking reproducible in the locked software environment.
4. As calibration posteriors concentrate at fixed reliability values, the quantile score converges to plug-in expected value. If
   \(\kappa\) also exceeds both absolute log-likelihood changes, clipping is inactive. This is a limiting result, not exact equality
   for a finite Beta posterior or finite Monte Carlo sample.

These statements do not prove that the policy is optimal over several tutoring turns. They also do not guarantee lower prediction
error when the evidence model is wrong. The held-out mismatch environments test that empirical question.

## Established and Project-Specific Parts

Bayesian odds updating, Beta-Bernoulli calibration, classification risk, and expected value of sample information are established
ideas. See Berger (1985), *Statistical Decision Theory and Bayesian Analysis*, second edition, Springer, and Raiffa and Schlaifer
(1961), *Applied Statistical Decision Theory*, Harvard University Press.

The project-specific design is their combination here: uncertain executable-probe reliability, a lower-quantile acquisition score,
bounded evidence updates, fixed-budget comparison, and tests in held-out evidence-failure environments. This combination is an
empirical method under evaluation, not new Bayesian theory.
