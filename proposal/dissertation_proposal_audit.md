# Technical, Architectural, and Mathematical Audit of `dissertation_proposal.tex`

Audited file: `proposal/dissertation_proposal.tex`  
Primary focus: Sections 2.4.1-2.4.4, with cross-checks against the architecture, evaluation plan, early traces, and implementation appendix.  
External sources checked include recent primary literature on LLM student simulation, Socratic RL tutors, cognitive/adaptive tutoring, knowledge tracing, and official LangGraph documentation. Links are listed at the end.

## Post-Revision Status Addendum

This audit report was written as a critical review of the original proposal. Since then, `proposal/dissertation_proposal.tex` has been revised to address the main high-severity mathematical and architectural issues. The original findings are retained below as review provenance; some line references and suggested replacements therefore refer to the pre-revision text.

### Issues Now Addressed in the Proposal

- The POMDP now uses a structured tutor control `u_t=(a_t,q_t,T_t,\ell_t,Z_{\text{tutor},t})`, so prompt load, target concept, realised text, and prompt difficulty are part of the causal intervention rather than hidden side effects of an abstract action label.
- The turn timing now separates pre-prompt state, post-prompt cognitive dynamics, evidence emission, and slower learning/misconception transitions.
- The Epistemic Tracker now uses a two-stage SMC update: particles are first projected through fast cognitive dynamics, weighted against evidence from the post-prompt/pre-learning state, and only then propagated through the slow learning transition. This fixes the earlier risk of using a post-learning state to explain the same probe outcome that caused learning.
- The evidence bundle now includes public response, execution telemetry, private probe outcome, guardrail event, and load-proxy signals. Prompt-load features remain part of `u_t`.
- The semantic likelihood now uses active evidence masks and log-space weighting rather than multiplying over every concept and misconception dimension.
- Weight entropy is no longer treated as posterior epistemic uncertainty; posterior marginal entropy and bandwidth dispersion are distinguished from SMC degeneracy diagnostics.
- CBFM is now framed as a simulated working-memory budget and overload-control model, not as direct measurement of human mental fatigue.
- CBFM now includes prompt-load features, bounded bandwidth dynamics, overload friction, an underchallenge diagnostic, soft overload probability, external load proxies, negative controls, calibration partitions, and held-out construct-utility tests.
- The evaluation plan now includes a decision-critical experiment gate for guardrail leakage, scratchpad falsification, CBFM construct utility, and policy complexity.
- The demo/ethics wording now avoids exposing raw hidden reasoning and treats human-in-the-loop deployment as future work requiring ethics and data-protection review.

### Residual Risks After Revision

- The strongest remaining scientific risk is construct validity: `B_t` and `F_t` will be credible only if they improve held-out prediction of external load proxies beyond prompt features and mastery/misconception controls.
- The novelty claim should stay narrow. The defensible claim is integration and falsifiable simulator control, not invention of cognitive load, fatigue modelling, student simulation, or RL tutoring.
- The automated classifiers that produce clarification/confusion/load proxy labels are now central to the validity story. Their agreement with expert transcript ratings must be reported, or those labels should be treated as exploratory.
- Tabular Q-learning is still likely to be underpowered if the discretised state abstraction grows. Q-table coverage and comparison against strong heuristic/contextual-bandit policies remain essential.
- Human efficacy remains outside the submitted simulation claim. Any language implying real-student learning improvement should remain conditional on future pilot evidence.

## 1. Mathematical Rigour

### Finding 1.1: The POMDP timing convention is internally inconsistent

**Severity: High**

The POMDP tuple is reasonable in spirit, but the turn indexing is not coherent across the formal definition, Bayesian update, algorithm, and cognitive variables.

- In Section 2.4.1, the transition is defined as `T(s_{t+1} | s_t, a_t)` at lines 359-363, but the observation function is written as `Z(o_t | s_t^{true}, T_t)` at lines 364-368. That implies the observation is generated from `s_t`, not `s_{t+1}`.
- In Section 2.4.3, the filter update at lines 407-412 uses `b_t(s')` after observing `o_{t-1}`, which implies the standard convention: action `a_{t-1}` moves the system into `s_t`, then observation `o_{t-1}` is emitted from `s_t`.
- Algorithm 1 at lines 735-744 updates `B_t` and `F_t` after generating the tutor prompt `T_t`, then generates `o_t`, then updates `K_{t+1}` and `M_{t+1}`. This is a different split: fast cognitive variables are updated inside the same turn, while knowledge is updated one step later.
- The footnote at line 363 says `B_{t+1}` and `F_{t+1}` update upon exposure to `a_{t+1}`, but the equations at lines 471-472 and Algorithm 1 update `B_t` from `T_t`.

This matters because a hostile reviewer can argue that the Markov state is undefined: are `B_t,F_t` pre-prompt capacity, post-prompt capacity, or the capacity available while answering?

**Fix:** introduce a two-stage turn state: pre-action latent state `s_t`, post-prompt working-memory state `\tilde{s}_t`, then next-turn state `s_{t+1}`. This preserves the intuition that cognitive load is fast and learning is slower.

```latex
\paragraph{Turn Timing Convention.}
At the beginning of turn $t$, the latent learner state is
\[
s_t = (\mathbf{K}_t,\mathbf{M}_t,B_t,F_t).
\]
The tutor selects a directive and target concept
\[
u_t=(a_t,q_t,T_t),
\]
where $T_t$ is the realised prompt generated from directive $a_t$.
Exposure to $T_t$ induces a fast cognitive update
\[
\tilde{B}_t = f_B(B_t,T_t,q_t), \qquad
\tilde{F}_t = f_F(\mathbf{K}_t,T_t,q_t),
\]
yielding the post-prompt state
\[
\tilde{s}_t=(\mathbf{K}_t,\mathbf{M}_t,\tilde{B}_t,\tilde{F}_t).
\]
The student response is then emitted as
\[
o_t \sim \mathcal{Z}(\cdot \mid \tilde{s}_t,u_t).
\]
Finally, learning and misconception transitions produce the next turn state:
\[
s_{t+1}\sim \mathcal{T}(\cdot \mid \tilde{s}_t,u_t,o_t).
\]
```

This formulation also resolves the ambiguity in whether observations are conditioned on `s_t` or `s_{t+1}`.

### Finding 1.2: The "action" is underspecified for the Markov property

**Severity: High**

The proposal defines actions as abstract directives such as `ask_counterfactual` and `present_analogy` (line 357), but the transition equations for `B_t`, `F_t`, and `K_t` depend on realised prompt properties: `L_t`, `N_{\text{concepts},t}`, `Z_{\text{tutor}}`, and the target concept `q_t` (lines 471-522, 542-565). Two prompts generated from the same directive can have very different length, difficulty, code span, and conceptual density. Therefore:

```latex
\mathcal{T}(s_{t+1}\mid s_t,a_t)
```

is not Markov unless the realised prompt features are included in the action or state.

**Fix:** define the environment-level action as a structured control tuple, while keeping the high-level directive as one component:

```latex
\[
u_t = (a_t, q_t, \ell_t, z_t, T_t),
\]
where $a_t \in \mathcal{A}$ is the pedagogical directive, $q_t$ is the targeted
concept, $\ell_t = (L_t,N_{\text{concepts},t},A_t,H_t^{\text{lex}})$ are prompt-load
features, and $z_t=Z_{\text{tutor},t}$ is the estimated prompt difficulty.
The transition and observation kernels are then
\[
\mathcal{T}(s_{t+1}\mid s_t,u_t), \qquad
\mathcal{Z}(o_t\mid s_{t+1},u_t).
\]
```

If the thesis wants the RL policy to choose only `a_t`, define a deterministic or stochastic renderer:

```latex
T_t \sim G_\psi(\cdot \mid a_t,q_t,H_t,\hat{s}_t),
```

and make clear that the POMDP transition marginalises over the renderer:

```latex
\mathcal{T}(s_{t+1}\mid s_t,a_t)
= \int \mathcal{T}(s_{t+1}\mid s_t,u_t) \, p_\psi(u_t\mid a_t,s_t)\,du_t.
```

Without this, the POMDP is closer to a history-dependent simulator than a Markov decision process.

### Finding 1.3: The reward function leaks simulator-only truth into policy learning

**Severity: Medium-High**

The reward in lines 369-377 uses true latent mastery gains:

```latex
R_t = \sum_i w_i(K^{true}_{i,t+1}-K^{true}_{i,t}) - w_fF_t - w_b(1-B_t)-w_wW_t.
```

This is acceptable as a simulator training reward, but the proposal then gives an expected reward under the tracker estimate using `\hat K_{t+1}-\hat K_t` (line 375). The issue is that `\hat K_{t+1}` is not available until after action execution and observation. In a POMDP, the policy chooses actions from the current belief; the environment returns the reward. The policy should not be described as optimising a reward it can compute before acting.

**Fix:** separate the reward used for simulator training from the deployment policy input.

```latex
\[
r_t^{\text{sim}} =
\sum_{i=1}^m w_i\Delta K^{\text{true}}_{i,t}
- w_f\tilde{F}_t
- w_b(1-\tilde{B}_t)
- w_w W_t,
\quad
\Delta K^{\text{true}}_{i,t}=K^{\text{true}}_{i,t+1}-K^{\text{true}}_{i,t}.
\]

The policy never observes $s_t^{\text{true}}$. It receives only
\[
x_t=\phi(b_t,H_t,C_t)
\]
and is trained in the simulator from sampled transitions
\[
(x_t,a_t,r_t^{\text{sim}},x_{t+1},done_t).
\]
Reported educational claims use held-out diagnostic outcomes rather than
$r_t^{\text{sim}}$.
```

### Finding 1.4: The belief update should use integrals and an observation bundle

**Severity: Medium**

The Bayesian update at lines 407-412 sums over `\mathcal{S}`, but the state contains continuous variables `K_t`, `B_t`, and `F_t`. A pure summation is formally incorrect for the hybrid state space. The proposal also says the observation stream contains natural language, execution outcomes, diagnostic probe results, and guardrail events (line 358), yet the update uses only `o_{t-1}`.

**Fix:** define a per-turn evidence object:

```latex
\[
e_t = (o_t, x_t^{\text{exec}}, y_t^{\text{probe}}, g_t^{\text{guard}}, \ell_t),
\]
```

then update with a hybrid integral/sum:

```latex
\[
b_{t+1}(s') =
\eta\,
\mathcal{Z}(e_t\mid s',u_t)
\int_{\mathcal{S}}
\mathcal{T}(s'\mid s,u_t,e_t)\,b_t(s)\,d s.
\]
```

For implementation, the particle approximation is sufficient:

```latex
s_{t+1}^{[i]} \sim \hat{\mathcal{T}}(\cdot\mid s_t^{[i]},u_t,e_t),
\qquad
\tilde{w}_{t+1}^{[i]} =
w_t^{[i]}\,\hat{\mathcal{Z}}(e_t\mid s_{t+1}^{[i]},u_t).
```

### Finding 1.5: The semantic likelihood is overconfident and dimensionally brittle

**Severity: High**

The likelihood at lines 423-434 multiplies over every concept and every misconception:

```latex
\prod_{j=1}^m(\cdots)\prod_{l=1}^k(\cdots)
```

This assumes conditional independence of all extracted indicators given the particle state. More importantly, it treats every unmentioned concept as evidence. If `m` and `k` are more than a handful, the product will underflow and punish particles for concepts that were never elicited in the turn. The text earlier claims "active-observation masking" (line 236), but the formal update does not implement it.

There is also notational confusion: `p^{obs}_{k,j,t-1}` uses `k` even though `k` already denotes the number of misconceptions, and `q^{obs}_{m,l,t-1}` uses `m` even though `m` denotes concept count.

**Fix:** use an active evidence set and log-likelihood. This is both mathematically cleaner and implementable.

```latex
\[
A_t^K \subseteq \{1,\ldots,m\}, \qquad A_t^M \subseteq \{1,\ldots,k\}
\]
denote concepts and misconceptions actively probed, mentioned, or implicated by
execution evidence at turn $t$. Let $\rho^K_{j,t}\in[0,1]$ and
$\rho^M_{\ell,t}\in[0,1]$ be calibrated classifier probabilities.

\[
\log \hat{\mathcal{Z}}(e_t\mid s^{[i]}_{t+1},u_t)
=
\sum_{j\in A_t^K}
\log \operatorname{Bern}\!\left(\rho^K_{j,t};
\epsilon_K + (1-2\epsilon_K)K^{[i]}_{j,t+1}\right)
+
\sum_{\ell\in A_t^M}
\log \operatorname{Bern}\!\left(\rho^M_{\ell,t};
\epsilon_M + (1-2\epsilon_M)M^{[i]}_{\ell,t+1}\right).
\]
\[
\log \tilde{w}_{t+1}^{[i]}
=
\log w_t^{[i]}+
\log\!\left((1-\alpha_{\text{smooth}})
\exp(\log \hat{\mathcal{Z}}_i)
\alpha_{\text{smooth}}\epsilon_{\text{smooth}}\right).
\]
```

Then normalise with the log-sum-exp trick.

### Finding 1.6: Entropy over particle weights is not epistemic uncertainty

**Severity: Medium**

At lines 462-466, `H(b_t)=-\sum_iw_i\log w_i` is called "joint Shannon entropy of the belief distribution." This is not the entropy of the student state posterior. It is primarily a degeneracy measure over particle weights. After systematic resampling, weights are uniform, so this entropy becomes high even if all particles occupy almost the same state. Conversely, low entropy can indicate weight collapse, not necessarily high certainty.

**Fix:** keep weight entropy as a degeneracy diagnostic, and define uncertainty for planning using posterior marginals or dispersion:

```latex
\[
H_{\text{weights}}(b_t) = -\sum_{i=1}^M w_t^{[i]}\log w_t^{[i]},
\]
\[
H_K(b_t)=\sum_{j=1}^m h\!\left(\sum_{i=1}^M w_t^{[i]}K^{[i]}_{j,t}\right),
\quad
h(p)=-p\log p-(1-p)\log(1-p),
\]
\[
\operatorname{Var}_B(b_t)=\sum_i w_t^{[i]}(B^{[i]}_t-\hat B_t)^2.
\]
```

Use `H_K`, misconception marginal entropy, or credible interval width to decide whether to issue diagnostic probes.

### Finding 1.7: The SMC convergence section overclaims

**Severity: Medium-High**

The proof sketch in lines 789-985 is useful but currently too strong.

- The "uniformly bounded weights" theorem depends on an assumption that previous weights are bounded by `c_scale` (line 828). That assumption is not guaranteed unless resampling or clipping enforces it every turn.
- The non-vanishing ESS bound at lines 835-870 may be strictly positive but can be practically meaningless when `\alpha_{\text{smooth}}\epsilon_{\text{smooth}}` is tiny.
- The systematic resampling theorem at lines 883-912 states `Var(N_j) <= W_j(1-W_j)`. The derivation actually gives `Var(N_j)=\delta_j(1-\delta_j) <= 1/4`, where `\delta_j` is the fractional part of `MW_j`. The stated comparison omits the factor `M`; the correct multinomial count variance is `MW_j(1-W_j)`.
- The `L_2` convergence theorem (lines 923-985) is acceptable as a finite-horizon particle-filter consistency claim, but the final sentence says smoothing prevents the error constant from diverging across turns (line 985). Standard bounds usually remain finite for fixed `t`; they do not remain uniformly small over arbitrary horizons without stronger mixing/forgetting assumptions.

**Fix:** weaken the theorem to a finite-horizon consistency statement.

```latex
\begin{theorem}[Finite-Horizon Consistency of the Epistemic Tracker]
Assume $\mathcal{S}$ is compact, the transition kernel is Feller, and the
smoothed likelihood satisfies
$0<\underline{z}\le \mathcal{Z}_{\text{smooth}}(e_t\mid s,u_t)\le \bar{z}<\infty$
for all $s,u_t,e_t$. For any fixed horizon $T<\infty$ and bounded measurable
test function $\phi$, the bootstrap particle filter with resampling satisfies,
for each $t\le T$,
\[
\mathbb{E}\left[\left|(p_t^M,\phi)-(p_t,\phi)\right|^2\right]
\le \frac{C_t}{M}\|\phi\|_\infty^2,
\]
where $C_t<\infty$ may grow with $t$ and with $\underline{z}^{-1}$.
\end{theorem}
```

Correct the resampling theorem:

```latex
\[
N_j\in\{\lfloor MW_j\rfloor,\lceil MW_j\rceil\},\qquad
\operatorname{Var}(N_j)=\delta_j(1-\delta_j)\le \frac14.
\]
\[
\operatorname{Var}_{\text{multi}}(N_j)=MW_j(1-W_j).
\]
```

### Finding 1.8: The bandwidth recurrence is mostly stable but underspecified at boundaries

**Severity: Medium**

The recurrence

```latex
B_t = B_{t-1}D_t+\mu_r(1-B_{t-1})
```

with `D_t=exp(-lambda_c N_t - lambda_l L_t)` does preserve `[0,1]` for finite `L_t`, `N_t>=0`, `D_t in (0,1]`, and `\mu_r in [0,1]`. The steady-state derivation at lines 489-497 is mostly correct. However:

- The claim that the stability condition is universally satisfied (line 493) is not true at the boundary `D=1,\mu_r=0`, where the coefficient is exactly `1`.
- The proposal says `L_t` is normalised at line 474 but uses raw character lengths in the traces at lines 1116-1149. That makes the parameter interpretation inconsistent.
- Recovery occurs in the same equation as prompt-induced depletion. That is defensible, but the model needs a time-step interpretation: recovery per tutor turn, not physiological recovery per minute.

**Fix:** state the invariance and stability constraints explicitly.

```latex
\[
D_t=\exp(-\lambda_cN_t-\lambda_lL_t-\lambda_fF_t),\qquad
B_t = \Pi_{[0,1]}\!\left(B_{t-1}D_t+\mu_r(1-B_{t-1})+\xi^B_t\right),
\]
where $\Pi_{[0,1]}(x)=\min(1,\max(0,x))$ and
$\xi^B_t\sim\mathcal{N}(0,\sigma_B^2)$ is optional process noise in simulation.
For deterministic analysis set $\xi^B_t=0$.
```

Add:

```latex
\[
\mu_r\in(0,1],\quad D_t\in(0,1],\quad L_t,N_t\ge0.
\]
```

If `L_t` is normalised, update the examples to use `\tilde L_t`. If raw length is used, rename it:

```latex
L_t^{\text{raw}}=\text{token or character count}, \qquad
\tilde L_t = \min(1,L_t^{\text{raw}}/L_{\max}).
```

### Finding 1.9: Cognitive Friction collapses ZPD, difficulty, and overload into one asymmetric penalty

**Severity: High for novelty/theory, Medium for implementation**

The friction equation at lines 518-522 is:

```latex
F_t = 1 - exp(-beta max(0, Z_tutor - Z_student,t)).
```

This captures "too hard" mismatch, but it ignores under-challenge. The text frames friction using "ZPD" but cites Bjork's desirable difficulties at line 518. Zone of Proximal Development is Vygotskian; Bjork supports desirable difficulty/retrieval practice, not ZPD. More importantly, desirable difficulty is not equivalent to friction. Some friction is productive; too much is overload; too little is boredom or shallow practice.

**Fix:** split friction into overload mismatch and underchallenge, or use a target difficulty band.

```latex
\[
\Delta_t = Z_{\text{tutor},t}-Z_{\text{student},t},\qquad
F_t^{+}=1-\exp(-\beta_+\max(0,\Delta_t-\delta_{\text{zpd}})),
\]
\[
F_t^{-}=1-\exp(-\beta_-\max(0,-\Delta_t-\delta_{\text{easy}})).
\]
```

Use `F_t^+` as overload friction and `F_t^-` as boredom/underchallenge. Then learning should peak inside the ZPD band:

```latex
\[
G_{\text{zpd}}(\Delta_t)=
\exp\left(-\frac{(\Delta_t-\delta^*)^2}{2\sigma_z^2}\right),
\]
\[
K_{i,t+1}=K_{i,t}
\alpha_{\text{learn}}(1-K_{i,t})B_tG_{\text{zpd}}(\Delta_t)
\prod_{j\in Parents(i)}K_{j,t}.
\]
```

This strengthens the theory because it stops the policy from being rewarded for trivially easy prompts that keep `F_t=0`.

### Finding 1.10: The private probe and misconception update are too deterministic

**Severity: Medium**

At lines 550-557, failing a private probe deterministically activates the targeted misconception if `B_t` is above threshold. That confounds misconception with ordinary error, slip, ambiguity, and probe difficulty. At lines 562-563, if `B_t < threshold`, pass probability is exactly zero, even though the formula includes a guessing probability.

**Fix:** use probabilistic misconception transitions and a soft capacity gate:

```latex
\[
g_B(B_t)=\sigma(\kappa_B(B_t-\theta_{\text{overload}})).
\]
\[
P(\text{Pass}(P_t)=1\mid s_t)
= c_{\text{guess}}
+ (1-c_{\text{guess}})
g_B(B_t)
\sigma\!\left(\alpha_{q_t}(K_{q_t,t}-d_{P_t})
-\gamma_{c_t}M_{c_t,t}\right).
\]
```

For misconceptions:

```latex
\[
P(M_{c_t,t+1}=0\mid M_{c_t,t}=1,\text{Pass},B_t)
= \sigma(\eta_0+\eta_BB_t+\eta_KK_{q_t,t}-\eta_FF_t),
\]
\[
P(M_{c_t,t+1}=1\mid M_{c_t,t}=0,\text{Fail},B_t)
= \sigma(\zeta_0+\zeta_d d_{P_t}-\zeta_BB_t).
\]
```

This keeps the simulator inspectable while acknowledging false positives and false negatives.

### Finding 1.11: The early traces do not validate the equations as strongly as claimed

**Severity: Medium**

The trace at lines 1111-1155 is illustrative, not validation. It also exposes two consistency issues:

- The proposal states `L_t` is a normalised prompt-load score, but the trace uses character lengths `120`, `550`, and `160`.
- In Turn 3, `B_3=0.32`, barely above `\theta_{\text{overload}}=0.3`, yet the narrative says the student passes the probe and jumps to full mastery `K=1.0` (lines 1147-1154). The learning update at lines 543-548 with `\alpha_{\text{learn}}\in(0,1)` cannot jump from `0.0` to `1.0` in one step unless `\alpha_{\text{learn}}=1`, which is excluded.

**Fix:** label these as didactic traces, not validation, and make values consistent with the update rule.

```latex
\paragraph{Illustrative Trace, Not Validation.}
The following trace is a hand-worked sanity check of the intended dynamics.
It is not used as evidence of empirical validity.
```

Then use a partial mastery increase, e.g. `K: 0.10 -> 0.42`, not `0.0 -> 1.0`.

## 2. Architectural & RL Feasibility

### Finding 2.1: LangGraph is a good orchestration fit, but the proposal conflates true simulator state and tracker state

**Severity: High**

The LangGraph architecture matches the problem structure: explicit graph control, typed state, checkpointed persistence, and interrupt/resume for future human-in-the-loop deployment are all aligned with official LangGraph capabilities. The proposal's use of a single-writer state buffer is architecturally sensible.

However, the current text says only the Epistemic Tracker writes to the central buffer (lines 356 and 602), while Algorithm 1 has the Simulated Student/StudentPolicy update the true state at lines 743-744. That is not merely an implementation detail; it changes the trust boundary.

**Recommended separation:**

- `EnvironmentState`: simulator-owned true latent state. Only the simulator environment updates it.
- `BeliefState`: tracker-owned particle belief and estimates. Only the Epistemic Tracker updates it.
- `GraphState`: orchestration state containing conversation, latest prompt, latest observation, telemetry, guardrail status, and references to the above.

This prevents the conceptual error of giving the tracker authority over the student's ground truth.

### Finding 2.2: The Pydantic schema is weaker than the mathematical state

**Severity: Medium**

The schema at lines 638-684 lacks:

- validators enforcing `0 <= B_t,F_t,K_i <= 1`;
- tracker estimates for `\hat B_t` and `\hat F_t`, even though policy baselines receive them at lines 392 and 1423;
- particle metadata, ESS, entropy/uncertainty, and confidence intervals;
- latest evidence bundle: execution outcome, probe result, guardrail event, prompt-load features.

**Drop-in schema direction:**

```python
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, confloat

UnitFloat = confloat(ge=0.0, le=1.0)

class CognitiveState(BaseModel):
    bandwidth: UnitFloat = 1.0
    overload_friction: UnitFloat = 0.0
    underchallenge: UnitFloat = 0.0

class TrackerEstimates(BaseModel):
    estimated_mastery: Dict[str, UnitFloat] = Field(default_factory=dict)
    misconception_marginals: Dict[str, UnitFloat] = Field(default_factory=dict)
    estimated_cognitive: CognitiveState = Field(default_factory=CognitiveState)
    effective_sample_size: float = 0.0
    posterior_uncertainty: Dict[str, float] = Field(default_factory=dict)

class ObservationEvidence(BaseModel):
    student_text: str = ""
    execution_passed: Optional[bool] = None
    probe_passed: Optional[bool] = None
    guardrail_leakage: bool = False
    prompt_load: Dict[str, float] = Field(default_factory=dict)
```

### Finding 2.3: The graph order is plausible but should be expressed as an environment step

**Severity: Medium**

The described cycle starts with the Epistemic Tracker, then Tutor, Guardrail, Student, then loops back (lines 601-627). This is fine after the first student turn, but the formal algorithm has to special-case `t=1` because no observation exists yet (lines 701-709).

For clarity, define the simulator as:

```text
Policy observes belief -> Tutor renders prompt -> Guardrail approves/rewrite -> Environment updates fast cognitive state -> Student emits response/probe/execution -> Tracker updates belief -> repeat.
```

This makes the Gym-style API cleaner:

```python
obs, reward, terminated, truncated, info = env.step(action)
```

Use Gymnasium's five-return convention rather than the older four-return Gym convention.

### Finding 2.4: The RL baseline plan is defensible but state explosion is not bounded

**Severity: High**

The baseline ordering is strong: static, heuristic, contextual bandit, Q-learning. The proposal correctly refuses to assume RL superiority (lines 396-404 and 1439-1448). That is one of the strongest parts of the design.

The weakness is that tabular Q-learning over a discretised belief feature vector can explode quickly. If there are:

- `m=8` concepts binned into 3 levels;
- `k=6` misconception bits;
- `B,F` binned into 3 levels each;
- 5 turn-count bins;

then the rough state count is:

```text
3^8 * 2^6 * 3^2 * 5 = 18,895,680 states
```

Tabular Q-learning will not meaningfully cover that space in an MSc-scale simulation unless `m,k` are tiny or the state abstraction is much more aggressive.

**Fix:** define a compact feature map:

```latex
\[
\phi(b_t)=(
\operatorname{bin}(\hat K_{q_t}),
\operatorname{bin}(\min_{j\in Parents(q_t)}\hat K_j),
\hat M_{c_t},
\operatorname{bin}(\hat B_t),
\operatorname{bin}(\hat F_t^+),
n_{\text{fail}},
n_{\text{hint}},
t_{\text{bin}})
\]
```

This keeps the Q-table interpretable and finite. Also report state coverage:

```latex
\[
\text{coverage}=\frac{|\{(x,a):N(x,a)>0\}|}{|\mathcal{X}||\mathcal{A}|}.
\]
```

If coverage is low, the Q-learning baseline should be replaced by fitted Q iteration, conservative Q-learning, or simply treated as underpowered.

### Finding 2.5: "Offline Q-learning" needs clearer terminology

**Severity: Medium**

The proposal uses "offline policy evaluation" to mean policy training and evaluation in a mathematical surrogate simulator before human deployment. That is acceptable informally, but in RL "offline RL" often means learning from a fixed logged dataset without environment interaction. The June 2026 contextual-bandit work on adaptive instructional policies specifically learns from logged interaction data with off-policy/counterfactual machinery.

**Fix:** rename the proposal's approach:

- "simulator-trained policy optimisation";
- "offline-from-human, online-in-simulator RL";
- "surrogate-environment policy sweep."

Reserve "offline RL/OPE" for logged data.

### Finding 2.6: Simulator interface should expose stochastic seeds and transition parameters

**Severity: Medium**

The simulator API at lines 387-393 is conceptually right. To support reproducibility and domain randomisation, `info_t` should include not only hidden true state but also:

- sampled student parameters: `alpha_learn`, `lambda_c`, `lambda_l`, `mu_r`, `beta`, slip/guess;
- prompt-load features;
- active evidence masks;
- renderer model ID and prompt template ID;
- random seed and scenario ID.

This will make ablation results auditable rather than anecdotal.

### Finding 2.7: Latency claims are optimistic and provider-bound

**Severity: Medium**

The scaling section at lines 1252-1264 correctly identifies sequential LLM latency as a bottleneck. The statement that `N=100` concurrent dialogues reduces wall-clock time from `O(N)` to approximately `O(1)` is too optimistic. With rate limits, token-per-minute caps, retry backoff, and provider queueing, throughput is bounded by external quotas. The claim should be phrased as "wall-clock time approaches the longest dialogue path only when provider concurrency and token budgets are non-binding."

Also, 30-50 sequential API calls per turn (line 1257) appears high for the graph shown. If this includes self-consistency, multiple classifiers, multiple guardrail rewrites, and evaluator voting, enumerate them. Otherwise a reviewer will see a mismatch between the architecture and the cost model.

### Finding 2.8: The local sandbox should not be called zero-trust

**Severity: Medium**

The sandbox section is outside the requested focus, but it affects feasibility. The local fallback uses AST filtering plus Python `exec` in a child process (lines 987-1107). This is useful for controlled synthetic code, but it is not a zero-trust sandbox for arbitrary hostile user code. The microVM option at lines 1542-1550 is the right production direction.

**Fix:** rephrase:

```latex
The local executor is a restricted validation harness for trusted or synthetic
student code during offline experiments. It is not relied upon as a production
security boundary for arbitrary human submissions; production deployment uses
ephemeral microVM isolation.
```

## 3. Novelty Assessment

### Overall novelty judgement

The proposal's strongest novelty is **not** "we invented cognitive load modelling" or "we measure mental fatigue." Those claims would be too broad. Cognitive load, affective tutoring, proactive support based on frustration/confusion, BKT/DRL adaptivity, and multimodal affective tutors already exist. The defensible novelty is narrower:

> an inspectable simulator control layer that couples prompt-load features, ZPD mismatch, private diagnostic probes, and particle-filtered belief estimates inside a LangGraph-orchestrated Socratic tutoring policy evaluation loop.

That is a credible niche if stated carefully and evaluated with ablations.

### Finding 3.1: `B_t` and `F_t` are novel as integration variables, not as psychological constructs

**Severity: High for thesis framing**

`B_t` resembles a bounded workload/resource variable. `F_t` resembles challenge-skill mismatch or overload relative to ZPD. These are not new constructs in cognitive science or intelligent tutoring. What may be new is their operationalisation inside an LLM student simulator and their use as policy features in a Socratic POMDP.

Recent SOTA context:

- Yuan et al. 2026 argue for explicit Epistemic State Specifications to prevent invalid LLM student simulation, but do not provide this exact dynamic bandwidth recurrence.
- Scarlatos et al. 2026 benchmark simulated students and show simple prompting is weak, supporting the need for stronger simulator validity checks.
- PEARL 2026 trains Socratic tutors with RL and a controllable student simulator that decouples latent cognitive states from response generation. This is close to the proposal and weakens any claim that latent-state Socratic RL itself is novel.
- Adaptive scaffolding work in 2026 compares BKT and DRL for cognitive engagement in ITS, showing that the BKT/DRL adaptivity idea is not new.
- Affective/multimodal tutoring work models emotions, confusion, frustration, and engagement; therefore "mental fatigue modelling" cannot be claimed as broadly novel.

**Recommended novelty claim:**

```latex
We do not claim to measure human mental fatigue directly. The contribution is
an inspectable Cognitive Bandwidth and Friction Modelling layer for simulated
LLM students: a bounded, ablatable set of control variables that operationalise
prompt load, prerequisite mismatch, and overload risk inside a POMDP tutoring
simulator. The novelty lies in coupling these variables to private diagnostic
probes, particle-filtered belief updates, and policy-level Socratic action
selection, then testing whether the coupling improves external behavioural
proxies under held-out tasks.
```

### Finding 3.2: The term "Mental Fatigue" should be avoided or heavily qualified

**Severity: High**

The proposal already says `B_t` is not direct human fatigue (lines 470-476 and 1591), which is good. Strengthen this consistently. "Mental fatigue" implies a latent psychophysiological construct requiring human telemetry, longitudinal effort data, or validated self-report instruments. The current model uses prompt features and simulator state. It is a **cognitive-load proxy** or **simulated working-memory budget**, not a fatigue measurement.

**Fix language:**

- Replace "measures mental fatigue" with "operationalises simulated per-turn processing pressure."
- Replace "available working memory capacity" with "available simulated working-memory budget."
- When discussing human deployment, say the variables require recalibration using self-report, latency, error patterns, and possibly multimodal signals.

### Finding 3.3: The novelty claim needs a stronger ablation logic

**Severity: Medium**

The proposal has a good ablation matrix at lines 1429-1437. To strengthen novelty, add targeted ablations for *which part* of CBFM matters:

- `B only`: set `F_t=0`, keep `B_t`.
- `F only`: set `B_t=1`, keep `F_t`.
- `Prompt-length only`: compute load from tokens only.
- `Semantic load`: use concept novelty/AST/dependency features.
- `Hard threshold vs soft overload`: compare indicator gate to sigmoid gate.
- `No recovery`: set `\mu_r=0`.
- `Human-like stochasticity`: add process noise to `B_t,F_t,K_t`.

The claim is much stronger if the thesis shows that concept/prerequisite-aware load beats raw token length and that soft gates beat brittle thresholds.

### Finding 3.4: Strengthen novelty against PEARL and ES-LLMS-style decoupling

**Severity: Medium**

PEARL already includes a controllable student simulator and pedagogically aligned RL. ES-LLMS-style architectures separate decision-making from natural-language rendering using BKT and rules. Therefore, "discrete pedagogical actions + RL + student simulator" is not enough.

Your differentiators should be:

- programming-specific execution evidence, not only dialogue;
- private scratchpad probes to reduce sycophancy;
- SMC over hybrid continuous-discrete belief state;
- cognitive load variables coupled to prompt-load and prerequisite mismatch;
- explicit falsification via ablations and expert transcript ratings.

### Finding 3.5: Add calibration targets for `B_t` and `F_t`

**Severity: Medium**

Novelty becomes more defensible if the variables are calibratable. Add a small calibration table:

| Latent variable | Observable proxy | Expected relation |
|---|---|---|
| `B_t` low | longer response latency, more "I am confused" markers, higher syntax/test failure, shorter or more avoidant responses | negative monotonic |
| `F_t^+` high | failure on target concept despite prerequisite success, clarification requests, expert-rated "too hard" prompt | positive monotonic |
| `F_t^-` high | correct but shallow responses, low diagnostic gain, boredom/off-task markers | positive monotonic |

Then preregister monotonicity tests:

```latex
\[
\Pr(\text{diagnostic fail}_{t+1}) =
\sigma(\eta_0+\eta_B(1-B_t)+\eta_FF_t^+ + \eta_K(1-K_{q_t,t})).
\]
```

This turns `B_t,F_t` from decorative telemetry into testable simulator variables.

## 4. Actionable Fixes

### Fix 4.1: Replace the POMDP formulation block

Use this as a replacement for the core formalism in Section 2.4.1.

```latex
\subsection{POMDP Formulation of Socratic Dialogue}
We model tutoring as a belief-state control problem over a hybrid latent
student state. At the beginning of turn $t$,
\[
s_t^{\mathrm{true}} =
(\mathbf{K}_t^{\mathrm{true}},\mathbf{M}_t^{\mathrm{true}},B_t,F_t),
\]
where $\mathbf{K}_t^{\mathrm{true}}\in[0,1]^m$,
$\mathbf{M}_t^{\mathrm{true}}\in\{0,1\}^k$, and
$B_t,F_t\in[0,1]$.

The policy observes a belief summary $x_t=\phi(b_t,H_t,C_t)$ rather than
$s_t^{\mathrm{true}}$. It selects a structured tutor control
\[
u_t=(a_t,q_t,T_t),
\]
where $a_t\in\mathcal{A}$ is a discrete Socratic directive, $q_t$ is the target
concept, and $T_t$ is the realised prompt produced by the generation layer.
The prompt induces fast cognitive dynamics
\[
\tilde{B}_t=f_B(B_t,T_t,q_t),\qquad
\tilde{F}_t=f_F(\mathbf{K}_t,T_t,q_t),
\]
and the student emits an evidence bundle
\[
e_t=(o_t,x_t^{\mathrm{exec}},y_t^{\mathrm{probe}},g_t^{\mathrm{guard}})
\sim \mathcal{Z}(\cdot\mid
\mathbf{K}_t,\mathbf{M}_t,\tilde B_t,\tilde F_t,u_t).
\]
The next latent state is sampled from
\[
s_{t+1}^{\mathrm{true}}\sim
\mathcal{T}(\cdot\mid
\mathbf{K}_t,\mathbf{M}_t,\tilde B_t,\tilde F_t,u_t,e_t).
\]
The tracker maintains $b_t(s)=P(s_t=s\mid e_{1:t-1},u_{1:t-1})$.
```

### Fix 4.2: Replace the belief update and particle weighting equations

```latex
\[
b_{t+1}(s') =
\eta\,\mathcal{Z}(e_t\mid s',u_t)
\int_{\mathcal{S}}\mathcal{T}(s'\mid s,u_t,e_t)b_t(s)\,ds.
\]

\[
s_{t+1}^{[i]} \sim \hat{\mathcal{T}}(\cdot\mid s_t^{[i]},u_t,e_t),
\qquad
\log \tilde{w}_{t+1}^{[i]}
=\log w_t^{[i]}+\log\hat{\mathcal{Z}}(e_t\mid s_{t+1}^{[i]},u_t).
\]

\[
w_{t+1}^{[i]}=
\frac{\exp(\log \tilde{w}_{t+1}^{[i]}-\operatorname{LSE}_j\log \tilde{w}_{t+1}^{[j]})}
{\sum_{r=1}^M \exp(\log \tilde{w}_{t+1}^{[r]}-\operatorname{LSE}_j\log \tilde{w}_{t+1}^{[j]})}.
\]
```

### Fix 4.3: Replace `\hat M_t = argmax` with marginal misconception estimates

```latex
\[
\hat K_{j,t}=\sum_{i=1}^M w_t^{[i]}K_{j,t}^{[i]},
\qquad
\hat p^M_{\ell,t}=\sum_{i=1}^M w_t^{[i]}M_{\ell,t}^{[i]},
\]
\[
\hat M_{\ell,t}=\mathbb{I}(\hat p^M_{\ell,t}\ge \tau_M).
\]
```

This avoids an intractable `argmax` over `2^k` misconception patterns.

### Fix 4.4: Replace CBFM with a bounded, soft-gated model

```latex
\[
\tilde L_t=
\omega_{\mathrm{tok}}\tilde T_t+
\omega_{\mathrm{code}}\tilde C_t+
\omega_{\mathrm{ast}}\tilde A_t+
\omega_{\mathrm{novel}}\tilde U_t+
\omega_{\mathrm{entropy}}\tilde H_t,
\qquad
\sum_r\omega_r=1,\quad \omega_r\ge0.
\]

\[
\Delta_t=Z_{\mathrm{tutor},t}-Z_{\mathrm{student},t},
\qquad
F_t^+=1-\exp(-\beta_+\max(0,\Delta_t-\delta_{\mathrm{zpd}})).
\]

\[
D_t=\exp(-\lambda_cN_t-\lambda_l\tilde L_t-\lambda_fF_t^+),
\qquad
B_t=\Pi_{[0,1]}\left(B_{t-1}D_t+\mu_r(1-B_{t-1})+\xi_t^B\right).
\]

\[
p_{\mathrm{overload}}(B_t)=\sigma(\kappa_B(\theta_{\mathrm{overload}}-B_t)).
\]
```

Then change the student policy from a hard indicator to a mixture:

```latex
\[
\pi_{\mathrm{student}}(\cdot)
=p_{\mathrm{overload}}(B_t)\pi_{\mathrm{overloaded}}(\cdot)
+(1-p_{\mathrm{overload}}(B_t))
\left[
p_{\mathrm{pass},t}\pi_{\mathrm{mastery}}(\cdot)
(1-p_{\mathrm{pass},t})\pi_{\mathrm{misconception}}(\cdot)
\right].
\]
```

### Fix 4.5: Correct the MIRT capacity gate

```latex
\[
p_{\mathrm{pass},t}
=c_{\mathrm{guess}}
+(1-c_{\mathrm{guess}})
g_B(B_t)
\sigma\!\left(
\alpha_{q_t}(K^{\mathrm{true}}_{q_t,t}-d_{P_t})
-\gamma_{c_t}M^{\mathrm{true}}_{c_t,t}
\right),
\]
\[
g_B(B_t)=\sigma(\kappa_B(B_t-\theta_{\mathrm{overload}})).
\]
```

If free-response coding has negligible guessing, set `c_guess=0` rather than multiplying by a hard overload indicator.

### Fix 4.6: Add process noise and forgetting to true state transitions

The current mastery update is monotonic and deterministic. That is too clean for simulation robustness and makes policy overfitting easier.

```latex
\[
K^{\mathrm{true}}_{i,t+1}=
\Pi_{[0,1]}\left[
K^{\mathrm{true}}_{i,t}
(1-K^{\mathrm{true}}_{i,t})\alpha_i B_tG_{\mathrm{zpd}}(\Delta_t)
\prod_{j\in Parents(i)}K^{\mathrm{true}}_{j,t}
-\rho_i K^{\mathrm{true}}_{i,t}
+\xi^K_{i,t}
\right],
\]
where $\rho_i\ge0$ is a small forgetting/interference rate and
$\xi^K_{i,t}$ is zero-mean process noise.
```

For root concepts, explicitly state:

```latex
\[
\prod_{j\in Parents(i)}K_{j,t}=1 \quad \text{when } Parents(i)=\emptyset.
\]
```

### Fix 4.7: Correct the policy-selection line in Algorithm 1

Current line 713 uses `B_{t-1},F_{t-1}`, which are true simulator variables. Replace with tracker estimates:

```latex
\State $a_t \gets \text{SelectPedagogicalDirective}(
\hat{\mathbf{K}}_t,\hat{\mathbf{M}}_t,\hat{B}_t,\hat{F}_t,
n_{\mathrm{fail}},W_{t-1})$
```

### Fix 4.8: Replace the "validation" wording for early traces

Current line 1111 says the traces validate the equations. Replace with:

```latex
The following hand-worked traces illustrate the intended qualitative behaviour
of the CBFM equations and scratchpad probing protocol. They are sanity checks,
not empirical validation; empirical support is provided only by the ablation
and held-out evaluation protocol in Chapter~\ref{ch:plan}.
```

### Fix 4.9: Add a source-of-truth architecture paragraph

```latex
\paragraph{Simulation State Ownership.}
During synthetic experiments, the simulator environment owns
$s_t^{\mathrm{true}}$ and is the only component permitted to mutate true
student variables. The Epistemic Tracker owns only the belief state $b_t$ and
commits estimated variables $(\hat{\mathbf K}_t,\hat{\mathbf M}_t,\hat B_t,
\hat F_t)$ to the graph state. The tutor policy receives only these estimates.
In human-in-the-loop deployment, $s_t^{\mathrm{true}}$ is absent; only the
belief state and observable evidence are retained.
```

### Fix 4.10: Strengthen the novelty paragraph

```latex
\paragraph{Novelty Boundary.}
CBFM is not presented as a validated psychometric measure of human mental
fatigue. It is a simulator-facing control model that makes assumptions about
prompt load, prerequisite mismatch, and overload explicit enough to ablate.
The novel contribution is the integration of these load variables with
private execution-gated diagnostic probes, SMC belief tracking, and constrained
Socratic policy selection in a programming-tutor simulator. The claim is
accepted only if CBFM improves external behavioural proxies under held-out
tasks relative to token-only, no-load, and heuristic-load ablations.
```

### Sources Checked

- Yuan et al. 2026, *Towards Valid Student Simulation with Large Language Models*: https://arxiv.org/abs/2601.05473
- Scarlatos et al. 2026, *Simulated Students in Tutoring Dialogues: Substance or Illusion?*: https://arxiv.org/abs/2601.04025
- Chang et al. 2026, *PEARL: Training Socratic Tutors with Pedagogically Aligned Reinforcement Learning*: https://arxiv.org/abs/2605.29582
- Tithi et al. 2026, *Adaptive Scaffolding for Cognitive Engagement in an Intelligent Tutoring System*: https://arxiv.org/abs/2602.07308
- Wei et al. 2026, *SLOW: Strategic Logical-inference Open Workspace for Cognitive Adaptation in AI Tutoring*: https://arxiv.org/abs/2603.28062
- Girard et al. 2026, *Counterfactual learning of new adaptive instructional policies using logged data*: https://arxiv.org/abs/2606.23015
- Berthon and van der Schaar 2025, *Language Bottleneck Models*: https://arxiv.org/abs/2506.16982
- Worden et al. 2026, *FoundationalASSIST*: https://arxiv.org/abs/2602.00070
- Ding et al. 2026, *ContextEcho*: https://arxiv.org/abs/2605.24279
- Zeng et al. 2025, *GraphMASAL*: https://arxiv.org/abs/2511.11035
- LangGraph Persistence docs: https://docs.langchain.com/oss/python/langgraph/persistence
- LangGraph Interrupts docs: https://docs.langchain.com/oss/python/langgraph/interrupts
- LangGraph Workflows and Agents docs: https://docs.langchain.com/oss/python/langgraph/workflows-agents
