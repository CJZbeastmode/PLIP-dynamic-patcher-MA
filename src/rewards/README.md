# Dynamic Patch RL — Reward Engine Documentation

This document provides a **complete and exhaustive overview** of the reward functions and reward engine presets used in the **Dynamic Patch RL** framework.  
It explains:
- All single-modal and multimodal reward modules  
- Their mathematical definitions  
- Their purpose and intuition  
- The structure of predefined reward engines  
- Recommended experimental groupings  
- How to choose a reward engine for testing  

---

# 1. Overview

The reward system is designed to be **modular**, **composable**, and **interpretable**.  
Each reward engine consists of:

- **Exactly one main zoom reward**  
  (InfoGain, CosineSimilarity, Entropy, TextAlign, etc.)

- **Zero or more extra rewards**  
  (TissuePresence, SemanticSurprise, PatchCost, etc.)

- **Structural penalties**  
  (DepthPenalty, StopPenalty, InvalidPatchPenalty, etc.)

A reward engine is implemented as:

```
r_total = Σ r_module
```

Each `RewardModule` contributes its own value depending on:
- parent patch
- children patches
- action (ZOOM or STOP)
- pyramid depth
- multimodal similarity
- and more

---

# 2. Reward Modules

This section lists all rewards implemented in `reward_module.py`.

---

## 2.1 Main Vision-Based Rewards

### **2.1.1 InfoGainReward**
Measures average cosine similarity between parent embedding `e_p` and child embeddings `e_c`.

\[
r_{	ext{info}} = lpha rac{1}{m}\sum_{i=1}^m \cos(e_p, e_c^{(i)})
\]

**Intuition:**  
Rewards zooming when children contain *new* or *different* information.

---

### **2.1.2 MaxDifferenceInfoReward**
Rewards maximum deviation between parent and any child.

\[
r_{	ext{info-diff}} = lpha (1 - \min_i \cos(e_p, e_c^{(i)}))
\]

**Intuition:**  
Rewards zooming when at least **one** child is drastically different.

---

### **2.1.3 MaxDifferenceRangeInfoReward**
Rewards large contrast among children's semantic similarity.

\[
r_{	ext{range}} = eta(\max_i s_i - \min_j s_j)
\]

**Intuition:**  
Encourages zooming if children vary significantly → heterogeneous detail.

---

### **2.1.4 CosineSimilarityReward**
Measures alignment between parent embedding and mean child embedding.

\[
r_{	ext{cos}} = eta\cdot \cos(e_p, ar e_c)
\]

**Intuition:**  
Rewards zooming when children refine/continue the current semantic trend.

---

## 2.2 Entropy-Based Rewards

### **2.2.1 EntropyGainReward**
\[
r_{	ext{entropy}} = \gamma(rac{1}{m}\sum_i H(C_i) - H(X))
\]

### **2.2.2 MaxEntropyGainReward**
\[
r_{	ext{max-entropy}} = \gamma(\max_i H(C_i) - H(X))
\]

**Intuition:**  
Rewards zooming into *visually complex* (non-blank) regions.

---

## 2.3 Multimodal (PLIP–FAISS) Rewards

These use the PLIP text encoder + FAISS-retrieved text embedding.

---

### **2.3.1 TextAlignReward**
Reward increases when children have higher multimodal similarity than parent.

\[
r_{	ext{text}} = \lambda(rac{1}{m}\sum_i s(C_i) - s(X))
\]

**Intuition:**  
Rewards zooming toward disease-relevant semantics.

---

### **2.3.2 TextAlignFixedEmbeddingReward**
Uses a *global* or *WSI-level* text embedding instead of patch-wise FAISS queries.

\[
s_{	ext{fixed}}(x) = \cos(e(x), e_{	ext{WSI}})
\]

\[
r_{	ext{text-fixed}} =
\lambda(rac{1}{m}\sum_i s_{	ext{fixed}}(C_i) - s_{	ext{fixed}}(X))
\]

**Intuition:**  
Simplified text alignment without querying FAISS per patch.

---

### **2.3.3 SemanticSurpriseReward**
Softmax-normalized difference:

\[
r_{	ext{surprise}} =
\delta\left|rac{1}{m}\sum_i 	ilde s_c^{(i)} - 	ilde s_pight|
\]

**Intuition:**  
Rewards zooming when children radically shift semantic meaning.

---

## 2.4 Tissue / Mask-Based Reward

### **TissuePresenceReward**
\[
r = \pm	au
\]

**Intuition:**  
Discourages zooming into blank areas.

---

## 2.5 Structural Penalties

### **PatchCost**
\[
r = -\lambda_{	ext{zoom}}
\]

### **DepthPenalty**
\[
r = -\eta(K - k)
\]

### **ZoomBudgetPenalty**
\[
r = 
egin{cases}
-\zeta, & 	ext{if zoom\_count} > B \
0
\end{cases}
\]

### **InvalidPatchPenalty**
\[
r = -M
\]

### **StopPenalty**
\[
r_{	ext{stop}} = s(X) - \lambda_{	ext{stop}}
\]

---

# 3. Reward Engine Groups

This section lists **all reward engines** and their conceptual grouping.

---

## 3.1 Group A — Minimal Engines
Used for core comparisons between:
- InfoGain
- MaxDifference
- Range
- CosSim
- Entropy
- TextAlign

Each engine contains:
- 1 main reward  
- PatchCost  
- StopPenalty  

This gives clean and fair comparisons.

---

## 3.2 Group B — Minimal + DepthPenalty
Same as Group A but includes DepthPenalty.

Purpose:
- Study the effect of depth control.

---

## 3.3 Group C — Conservative Exploration
Low zoom, aggressive penalization.
Used when:
- Minimize zoom cost  
- Prioritize high-confidence regions  

---

## 3.4 Group D — Aggressive Exploration
High zoom, low cost.  
Used when:
- Exploring texture-heavy WSIs  
- Searching deep variations  

---

## 3.5 Group E — Multimodal-Focused
Prioritizes FAISS + PLIP semantic signals.

---

## 3.6 Group F — Entropy-Driven
Used for:
- Patch-level texture exploration  
- Homogeneity detection  

---

## 3.7 Group G — Balanced / Production-Level
Mix of all modalities:
- InfoGain  
- TextAlign  
- TissuePresence  
- Penalties  
- SemanticSurprise (optional)

Used for final production experiments.

---

## 3.8 Group H — Special Purpose

### Examples:
- Depth-limited engines  
- High-quality safety engines  
- Exploration-only engines  

---

# 4. Recommended Testing Strategy

You should test **only the necessary minimal sets**.

### **1. Compare main reward families**
Use Group A:
- InfoGain
- MaxDifference
- Range
- CosSim
- Entropy
- TextAlign
- TextAlignFixed

### **2. Compare DepthPenalty influence**
Use Group B:
- InfoGain_only + DepthPenalty  
- TextAlign_only + DepthPenalty  
- Entropy_only + DepthPenalty  

### **3. Optional exploration**
Multimodal engines vs. entropy engines.

### **4. Production-level**
Balanced engines.

---

# 5. Directory Structure for Reward Experiments

```
rewards/
    reward_module.py
    rewards.py
    README_REWARDS.md
    engines/
        minimal/
        depth/
        conservative/
        aggressive/
        multimodal/
        entropy/
        balanced/
        special/
```

---

# 6. Summary

This reward architecture enables:
- Clear ablations  
- Modular construction  
- Direct main reward comparison  
- Scalable extensions  

You now have:
- Full documentation
- Proper grouping  
- Clean mathematical definitions  
- A structured testing methodology  

