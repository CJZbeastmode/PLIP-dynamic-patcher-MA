## Reward Parameter Analysis Report
**Date:** 5 December 2025

---

## 📊 Reward Value Range Analysis

### **Base Reward Ranges (before weighting):**

| Reward Module | Typical Range | Expected Behavior |
|---------------|---------------|-------------------|
| **InfoGainReward** | [0.0, 1.0] | 1.0 - mean(cos_sim), dissimilarity |
| **MaxDifferenceInfoReward** | [0.0, 2.0] | 1.0 - min(cos_sim), worst match |
| **MaxDifferenceRangeInfoReward** | [0.0, 2.0] | max(cos) - min(cos), spread |
| **CosineSimilarityReward** | [-1.0, 1.0] | Direct cosine similarity |
| **EntropyGainReward** | [-5.5, 5.5] | Entropy difference (bits) |
| **MaxEntropyGainReward** | [-5.5, 5.5] | Max entropy diff (bits) |
| **TextAlignReward** | [-1.0, 1.0] | FAISS score difference |
| **TissuePresenceReward** | {-1.0, 1.0} | Binary tissue/blank |
| **PatchCost** | {-cost, 0} | Fixed penalty per zoom |
| **StopPenalty** | [-penalty, 1.0-penalty] | s_parent - penalty |
| **DepthPenalty** | [-weight*depth, 0] | Scales with depth |

---

## 🔍 SECTION 1 Analysis: Minimal Engines

### **Current Parameters:**
```python
weight=5.0, PatchCost=0.002, StopPenalty=0.5
```

### **Expected Reward Calculation (ZOOM action):**

**For InfoGain-based:**
```
InfoGainReward: 5.0 * (1.0 - 0.8) = 1.0  (if mean cos_sim = 0.8)
PatchCost:      -0.002
Total ZOOM:     ~0.998

STOP reward:    s_parent - 0.5 ≈ 0.0 - 0.5 = -0.5
```

### **Problem 1: FAISS Returns 0** 🔴

**StopPenalty computation:**
```python
s_parent = embedder.plip_sim(ep)  # FAISS lookup
return s_parent - penalty
```

**If FAISS is broken or not initialized:**
- `s_parent = 0.0` (no FAISS results)
- `StopPenalty = 0.0 - 0.5 = -0.5`
- **STOP always gets -0.5 penalty**

**This explains user's observation: "models don't zoom at all"**

When FAISS returns 0:
- ZOOM reward: ~1.0 (positive)
- STOP reward: -0.5 (negative)
- Model **should** prefer ZOOM

But in practice, if rewards are normalized poorly or model is untrained, may still pick STOP.

---

### **Problem 2: Reward Imbalance** 🟡

**Comparing reward magnitudes:**

| Action | Component | Value |
|--------|-----------|-------|
| ZOOM | InfoGainReward | 0.0 to 5.0 |
| ZOOM | PatchCost | -0.002 |
| ZOOM | **Total** | **-0.002 to 5.0** |
| STOP | StopPenalty | -0.5 to 0.5 |

**Analysis:**
- ZOOM reward dominates (5x larger magnitude)
- PatchCost (0.002) is negligible compared to InfoGain (5.0)
- StopPenalty is smaller but comparable

**Consequence:**
- If InfoGain > 0.1, ZOOM always preferred
- Model has little reason to STOP unless InfoGain very low
- May cause excessive zooming

---

### **Problem 3: Section 2 Has DOUBLE StopPenalty** 🔴

**infogain_only_depth:**
```python
StopPenalty(penalty=-0.5, embedder=embedder),  # First one
StopPenalty(penalty=0.25, embedder=embedder),  # Second one!
```

**This is a BUG!**
- Two StopPenalty modules will both execute
- Total STOP reward = (s_parent - (-0.5)) + (s_parent - 0.25)
                    = 2*s_parent - 0.25 + 0.5
                    = 2*s_parent + 0.25

**If FAISS returns 0:**
- STOP reward = 0 + 0.25 = **0.25** (positive!)
- This **encourages** stopping, opposite of intended

**This is INCORRECT - should only have ONE StopPenalty**

---

## 🔧 Parameter Issues by Engine

### **SECTION 1: Minimal Engines** ⚠️

**Issue:** Rely on FAISS for StopPenalty
- If FAISS broken: STOP always penalized
- If FAISS=0: May cause training issues

**Recommendation:**
- Remove FAISS dependency from StopPenalty
- Use simple constant penalty

**Fix:**
```python
infogain_only = RewardEngine([
    InfoGainReward(weight=5.0, embedder=embedder),
    PatchCost(cost=0.01),           # Increase from 0.002
    StopPenalty(penalty=1.0),       # Remove embedder dependency
])
```

---

### **SECTION 2: Minimal + Depth** 🔴

**Critical Bug:** `infogain_only_depth` has **TWO StopPenalty modules**

```python
infogain_only_depth = RewardEngine([
    InfoGainReward(weight=1.0, embedder=embedder),
    PatchCost(cost=0.01),
    StopPenalty(penalty=-0.5, embedder=embedder),  # ❌ First penalty
    StopPenalty(penalty=0.25, embedder=embedder),  # ❌ Second penalty
])
```

**Must fix - remove duplicate:**
```python
infogain_only_depth = RewardEngine([
    InfoGainReward(weight=3.0, embedder=embedder),
    PatchCost(cost=0.005),
    StopPenalty(penalty=0.5, embedder=embedder),   # Single penalty
    DepthPenalty(weight=0.01),                     # Add missing DepthPenalty
])
```

---

### **SECTION 3: Ablation Engines** ⚠️

**Issue:** No PatchCost or StopPenalty
- Model has no incentive to ever STOP
- Will zoom to level 0 always
- Wastes computation

**Example:**
```python
ablation_infogain_only = RewardEngine([
    InfoGainReward(weight=10, embedder=embedder),
    # Missing PatchCost and StopPenalty!
])
```

**Expected behavior:**
- ZOOM: reward = 10 * (1 - cos_sim) ∈ [0, 10]
- STOP: reward = 0
- **Always ZOOMs** (never stops)

**Recommendation:**
Add at minimum a PatchCost to encourage stopping:
```python
ablation_infogain_only = RewardEngine([
    InfoGainReward(weight=10, embedder=embedder),
    PatchCost(cost=0.01),  # Add this
])
```

---

### **SECTION 7: Special Engines** ⚠️

**fast_exploration:**
```python
fast_exploration = RewardEngine([
    MaxDifferenceInfoReward(weight=10, embedder=embedder),
    PatchCost(cost=0.0001),  # Extremely small!
])
```

**Issue:** PatchCost = 0.0001 is negligible
- Reward range: [0, 10]
- Cost: 0.0001
- Cost is 0.001% of reward magnitude
- Effectively no cost at all

**Consequence:** Will zoom excessively

**Fix:** Increase PatchCost to at least 0.01

---

## 📊 Recommended Parameter Guidelines

### **General Principles:**

1. **Reward Magnitude Balance:**
   - Main reward (InfoGain, Entropy, etc.): 1-10
   - PatchCost: 0.01-0.1 (1-10% of main reward)
   - StopPenalty: 0.1-1.0 (10-100% of main reward)
   - DepthPenalty: 0.001-0.01 per level

2. **Avoid FAISS Dependency:**
   - Don't use `embedder` in StopPenalty unless FAISS guaranteed
   - Use constant penalties instead

3. **Every Engine Should Have:**
   - At least one main reward (InfoGain, Entropy, TextAlign)
   - PatchCost to limit zooming
   - StopPenalty OR DepthPenalty to control depth

4. **Sign Conventions:**
   - Positive StopPenalty = penalty for stopping = encourages zooming
   - Negative StopPenalty = reward for stopping = encourages stopping
   - **Never mix both in same engine!**

---

## 🔧 Specific Fix Recommendations

### **Fix 1: Remove FAISS from StopPenalty** [HIGH PRIORITY]

**Problem:** FAISS may return 0, causing unpredictable behavior

**Solution:** Use constant penalty
```python
# BEFORE:
StopPenalty(penalty=0.5, embedder=embedder)

# AFTER:
StopPenalty(penalty=0.5)  # Remove embedder
```

**But StopPenalty is implemented to require FAISS!**

Looking at the code:
```python
class StopPenalty(RewardModule):
    def compute(self, action=None, parent_patch=None, **kwargs):
        if action != 0 or parent_patch is None:
            return 0.0
        
        ep = self.embedder.img_emb(parent_patch)
        s_parent = self.embedder.plip_sim(ep)  # Requires FAISS!
        return float(s_parent - self.penalty)
```

**This is a design flaw** - StopPenalty shouldn't depend on FAISS.

**Better design:**
```python
class SimpleStopPenalty(RewardModule):
    """Simple constant penalty for stopping."""
    def __init__(self, penalty=0.5):
        self.penalty = penalty
    
    def compute(self, action=None, **kwargs):
        if action != 0:
            return 0.0
        return -self.penalty  # Negative = penalty
```

---

### **Fix 2: Remove Duplicate StopPenalty** [CRITICAL]

**File:** `src/rewards/rewards.py`
**Line:** ~113

```python
# BEFORE:
infogain_only_depth = RewardEngine([
    InfoGainReward(weight=1.0, embedder=embedder),
    PatchCost(cost=0.01),
    StopPenalty(penalty=-0.5, embedder=embedder),
    StopPenalty(penalty=0.25, embedder=embedder),  # ❌ REMOVE THIS
])

# AFTER:
infogain_only_depth = RewardEngine([
    InfoGainReward(weight=3.0, embedder=embedder),
    PatchCost(cost=0.005),
    StopPenalty(penalty=0.5, embedder=embedder),
    DepthPenalty(weight=0.01),
])
```

---

### **Fix 3: Add PatchCost to Ablation Engines** [MEDIUM PRIORITY]

All ablation engines should have at minimum a PatchCost:

```python
ablation_infogain_only = RewardEngine([
    InfoGainReward(weight=10, embedder=embedder),
    PatchCost(cost=0.01),  # ADD THIS
])
```

---

### **Fix 4: Rebalance Section 1 Parameters** [MEDIUM PRIORITY]

Current parameters too aggressive for zooming:

```python
# BEFORE:
weight=5.0, PatchCost=0.002, StopPenalty=0.5

# AFTER (more balanced):
weight=3.0, PatchCost=0.01, StopPenalty=0.3
```

This creates:
- ZOOM reward: 0-3.0 (InfoGain) - 0.01 (cost) = -0.01 to 2.99
- STOP reward: -0.3 to 0.7 (if FAISS works)
- More balanced trade-off

---

## 📋 Summary of Issues

| Issue | Severity | Engines Affected | Fix Required |
|-------|----------|------------------|--------------|
| **Duplicate StopPenalty** | 🔴 CRITICAL | infogain_only_depth | Remove one |
| **FAISS dependency** | 🟡 HIGH | All Section 1, 2 | Remove embedder param |
| **Missing PatchCost** | 🟡 MEDIUM | All ablation engines | Add PatchCost |
| **Negligible PatchCost** | 🟢 LOW | fast_exploration | Increase value |
| **Reward imbalance** | 🟢 LOW | Section 1 | Rebalance weights |

---

## ✅ Validated Engines (Good Parameters)

These engines have appropriate parameter balance:

- `balanced_standard`
- `balanced_comprehensive`
- `balanced_diversity`
- `vision_contrast`
- `vision_structural`
- `quality_assurance`

---

## 🎯 Action Items

### **Critical (Must Fix Before Training):**
1. ✅ Remove duplicate StopPenalty from `infogain_only_depth`
2. ✅ Verify FAISS is initialized properly

### **High Priority (Should Fix):**
3. ✅ Create SimpleStopPenalty class without FAISS dependency
4. ✅ Update Section 1 engines to use SimpleStopPenalty
5. ✅ Rebalance Section 1 weights

### **Medium Priority (Nice to Have):**
6. ⏭️ Add PatchCost to all ablation engines
7. ⏭️ Increase PatchCost in fast_exploration
8. ⏭️ Add DepthPenalty to Section 2 engines

---

**End of Parameter Analysis**
