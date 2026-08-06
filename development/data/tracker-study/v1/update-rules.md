# Bayesian Tracker Update Rules

## Scope

This study will compare five estimators inside a declared simulator. The hidden variable is binary mastery, $K_t \in \{0, 1\}$ \(K_t \in \{0,1\}\). The estimates are probabilities about that simulated variable; they are not measurements of a person's knowledge.

Every tracker receives the same observation record, \`(channel, category, confidence)\`. A tracker may deliberately ignore a field. This is part of the baseline definition, not missing data.

## Timing

Evidence corrects the belief about the current state first. Learning and forgetting then produce the prior for the next turn:

$$
\begin{aligned}
p_t^+ &= P(K_t=1 \mid z_t, c_t) \\[0.5em]
p_{t+1} &= p_t^+(1 - f) + (1 - p_t^+)l
\end{aligned}
$$

\[
p_t^+ = P(K_t=1 \mid z_t,c_t),
\qquad
p_{t+1}=p_t^+(1-f)+(1-p_t^+)l,
\]

where \(l\) is the learning probability and \(f\) is the forgetting probability. This ordering prevents a transition caused by the current interaction from being used to explain evidence generated before that transition.

## Ordinary Channel-Aware Update

For channel \(c\), observed category \(z\), and state \(k\), the frozen matrix gives

$$O_c(z, k) = P(Z = z \mid K = k, c)$$

\[
O_c(z,k)=P(Z=z \mid K=k,c).
\]

Bayes' rule in odds form is

$$\frac{p_t^+}{1 - p_t^+} = \frac{p_t}{1 - p_t} \cdot \frac{O_c(z, 1)}{O_c(z, 0)}$$

\[
\frac{p_t^+}{1-p_t^+}
=
\frac{p_t}{1-p_t}
\frac{O_c(z,1)}{O_c(z,0)}.
\]

Taking logarithms gives the implemented additive update:

$$\operatorname{logit}(p_t^+) = \operatorname{logit}(p_t) + \log \frac{O_c(z, 1)}{O_c(z, 0)}$$

\[
\operatorname{logit}(p_t^+)
=
\operatorname{logit}(p_t)
+
\log\frac{O_c(z,1)}{O_c(z,0)}.
\]

This is an established Bayesian identity. Applying it to distinct dialogue, executable-probe, and self-explanation channels is the project-specific modelling choice.

## Bounded Channel-Aware Update

The robust variant limits one observation's influence and then applies a fixed channel trust weight:

\[
\operatorname{logit}(p_t^+)
=
\operatorname{logit}(p_t)
+
\eta_c\,\operatorname{clip}
\left(
\log\frac{O_c(z,1)}{O_c(z,0)},
-\kappa,
\kappa
\right),
\]

with \(0 \leq \eta_c \leq 1\) and \(\kappa>0\). Therefore,

\[
\left|
\operatorname{logit}(p_t^+) - \operatorname{logit}(p_t)
\right|
\leq \eta_c\kappa.
\]

The bound follows directly from the clipping interval and multiplication by a non-negative trust weight. It is a guarantee about update size, not a guarantee of better predictions. The clipping and trust design is project-specific.

## Baselines

The last-observation baseline maps supportive evidence to \(1-\epsilon\), opposing evidence to \(\epsilon\), and ignores history. Hard BKT uses one fixed slip and guess pair for every channel. Both ignore confidence.

The legacy fractional-Bernoulli baseline converts category and confidence to a soft target \(\rho \in [0,1]\), then evaluates

\[
q^{\rho}(1-q)^{1-\rho}.
\]

This is retained because it reflects the proposal's earlier soft-label update. It is a pseudo-likelihood, not the probability mass of a fractional Bernoulli observation. It must not be described as exact Bayesian filtering.

The ordinary and bounded channel-aware trackers use the observed category and channel matrix. Confidence remains in their shared input record but is not used by these two estimators; channel reliability is already represented by \(O_c\). This avoids counting the same reliability judgement twice.

## Properties and Assumptions

All observation probabilities are strictly positive, so every likelihood ratio is finite. Probabilities are clipped only to the frozen numerical floor before conversion to log-odds and after conversion back.

If \(O_c(z,1)=O_c(z,0)\), the log-likelihood ratio is zero and neither channel-aware tracker corrects the prior. If \(O_c(z,1)>O_c(z,0)\), ordinary Bayes raises the posterior; if the inequality is reversed, it lowers it. Setting \(\eta_c=1\) and disabling clipping recovers the ordinary update exactly.

The simulator assumes observations are conditionally independent given the current mastery state and channel. That assumption is intentionally violated by the contradictory stress condition. Results under that condition measure brittleness to dependence; they do not repair or validate the independence assumption.
