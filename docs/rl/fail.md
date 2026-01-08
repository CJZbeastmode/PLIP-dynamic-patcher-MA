# Conclusion and Discussion

## Overview

This work investigated reinforcement learning (RL) for hierarchical patch selection in whole-slide images (WSIs), framed as a sequential decision problem with two actions: **STOP** or **ZOOM**. The objective was to determine whether an RL agent (Actor–Critic / A2C) could outperform greedy and supervised baselines by learning non-myopic zooming strategies under different patch scoring functions.

Across extensive experiments, RL did not consistently outperform greedy or supervised approaches. This section explains why this outcome is expected, why it does not indicate a failure of the implementation, and what it reveals about the structure of the problem itself.

---

## RL Did Not Fail — It Converged Correctly

A critical observation is that the RL training loops behaved as expected:

- Stable convergence of policy and value networks
- Decreasing entropy over training
- No divergence, collapse, or oscillatory instability
- Consistent action probabilities in later epochs
- Predictable plateauing of episode returns

The emergence of policies dominated by either STOP or ZOOM actions is therefore not a training failure, but the optimal solution under the given reward structure.

---

## Dominance of Reward Structure

All patch score modules used in this work share a key structural property:

> STOP and ZOOM scores are computed locally at the same state using the same embedding model, differing only by spatial aggregation.

This induces a locally separable decision boundary and removes most long-horizon credit assignment from the problem. As a result, RL is structurally constrained to learn policies that resemble local classifiers.

---

## Conservative Scores: TextAlignScore

### Behavior

TextAlignScore consistently favors STOP decisions at coarse pyramid levels. This is due to:

1. **Context aggregation bias**  
   Larger patches contain more semantic context and therefore receive higher text–image similarity scores.

2. **Mean aggregation over children**  
   Zoom scores are averaged across child patches, suppressing high-similarity outliers.

3. **Small score margins**  
   The difference `s_zoom − s_stop` is often close to zero or negative.

### Effect on RL

- Weak advantage signals
- Early termination becomes optimal
- Exploration is suppressed
- Policies converge to conservative STOP behavior

### Interpretation

RL correctly identifies that zooming rarely improves expected reward. Greedy and supervised baselines reach the same conclusion via direct score comparison, explaining the lack of RL advantage.

---

## Aggressive Scores: ImgSimScore

### Behavior

ImgSimScore rewards dissimilarity between parent and child patches, effectively encoding an information-gain prior. This leads to:

- Large positive ZOOM rewards
- High reward variance
- Strong bias toward deeper zooming

### Effect on RL

- Policies collapse into near-always-ZOOM behavior
- STOP actions become dominated
- Depth penalties only limit recursion depth, not decision quality

### Interpretation

RL optimizes the reward as defined, not semantic relevance. Since the score intrinsically favors zooming, RL converges to aggressive policies identical to greedy inference.

---

## Why RL Matches Greedy and Supervised Baselines

In this formulation:

- Rewards are local
- STOP and ZOOM are evaluated at the same state
- No delayed rewards exist
- Episodes terminate immediately on STOP

Under these conditions, the problem reduces to binary classification:

> If the optimal action can be determined from local features alone, RL collapses to supervised decision-making.

This explains why greedy thresholding and supervised classifiers match RL performance with lower complexity and faster convergence.

---

## Lack of Long-Horizon Credit Assignment

Although the environment is sequential, rewards do not meaningfully propagate across time:

- Zoom rewards do not depend on future outcomes
- STOP ends the episode
- No delayed benefit exists for preparatory actions

Consequently:

- The value function approximates immediate reward
- Bootstrapping adds little information
- Actor–Critic degenerates into a noisy classifier

---

## Why Early Training Appeared Worse

Initial training runs showed instability due to:

- Unscaled reward magnitudes
- Asymmetric STOP/ZOOM rewards
- Implicit action bias from clipping
- Inconsistent depth penalties

Once rewards were refactored to be symmetric and action-relative, training stabilized and converged faster to the same solutions.

---

## Scientific Interpretation (Negative Result)

The central finding can be stated precisely:

> For hierarchical patch selection driven by local semantic scores, reinforcement learning does not provide a structural advantage over greedy or supervised approaches.

This is not a limitation of RL itself, but a consequence of the reward design.

---

## Implications for Future Work

To justify RL in this setting, at least one of the following is required:

1. Non-local rewards (e.g., diagnostic correctness)
2. Delayed supervision at deeper levels
3. Global budget constraints
4. Cross-region dependency between decisions

Without these, RL will correctly converge to greedy-equivalent policies.

---

## Final Takeaway

RL behaved exactly as expected and revealed that it is unnecessary for this reward structure. This negative result is scientifically meaningful and supports the use of simpler, more interpretable baselines.

**RL did not fail — it proved it was the wrong tool for this formulation.**
