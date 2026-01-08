## Semantic Analysis: Training vs Inference Logic
**Date:** 5 December 2025

---

## 🎯 Executive Summary

**Status: ⚠️ SEMANTIC MISMATCH DETECTED**

The training loop and inference loop implement **different exploration strategies**, which creates a **train-test discrepancy** that may limit model effectiveness.

---

## 📊 Comparison Table

| Aspect | Training | Inference | Match? |
|--------|----------|-----------|--------|
| **Exploration** | Single-path (child 0 only) | Multi-path (all children) | ❌ |
| **State representation** | 515-dim (coords + embedding) | 515-dim (coords + embedding) | ✅ |
| **Action space** | {STOP=0, ZOOM=1} | {STOP=0, ZOOM=1} | ✅ |
| **Reward signal** | Immediate reward per step | No reward (greedy inference) | ⚠️ |
| **Episode termination** | STOP or level 0 or max_steps | STOP or level 0 | ✅ |
| **Child selection** | Hardcoded (child 0) | All children explored | ❌ |

---

## 🔍 Deep Dive: Training Loop

### **Algorithm: On-Policy Actor-Critic (A2C/REINFORCE variant)**

```python
def run_episode(env, model, optimizer):
    # 1. COLLECT TRAJECTORY
    state = env.reset()  # Random position at max_level
    done = False
    
    trajectory = []
    while not done:
        # Sample action from policy
        logits, value = model(state)
        action ~ Categorical(logits)  # Stochastic sampling
        
        # Take action in environment
        next_state, reward, done = env.step(action)
        
        # Store transition
        trajectory.append((state, action, reward, value))
        state = next_state
    
    # 2. COMPUTE RETURNS (Monte Carlo)
    G_t = Σ γ^k * r_{t+k}  # Discounted cumulative reward
    
    # 3. UPDATE POLICY
    advantages = (G_t - V(s_t))  # Advantage = how much better than expected
    policy_loss = -Σ log π(a_t|s_t) * advantage_t
    value_loss = MSE(V(s_t), G_t)
    
    loss = policy_loss + 0.5 * value_loss - 0.1 * entropy
    optimizer.step()
```

### **What the Model Learns:**

The model learns to **maximize expected cumulative reward** by:
1. **Stopping** when zooming would give low reward (blank patches, noise)
2. **Zooming** when children have high information gain (tissue structure)

### **Critical Constraint: Single-Path Exploration**

```python
# In env.step(action=1):  # ZOOM
child_patches = extract_all_children()  # 4 or 16 children
reward = compute_reward(parent, child_patches)  # Uses all children

# But only follows ONE child:
idx = 0  # Hardcoded!
self.curr_x, self.curr_y = child_coords[idx]
state = encode_state(child_patches[idx])
```

**Implications:**
- Model sees reward based on ALL children
- But only experiences trajectory through child 0
- Model does NOT learn which child is best to follow
- Model only learns ZOOM vs STOP at current position

---

## 🔍 Deep Dive: Inference Loop

### **Algorithm: Recursive Greedy Tree Search**

```python
def infer_zoom(env, model, level, x, y, max_depth=10):
    patch = get_patch(level, x, y)
    state = encode_state(patch, level, x, y)
    
    # Greedy action selection (no sampling)
    action = argmax(model(state))  # Deterministic!
    
    if action == STOP:
        kept.append(patch)
        return kept, discarded
    
    # If ZOOM: explore ALL children recursively
    discarded.append(patch)
    for each child in get_all_children(level):
        kept_child, disc_child = infer_zoom(model, child)
        kept.extend(kept_child)
        discarded.extend(disc_child)
    
    return kept, discarded
```

### **What Actually Happens:**

The inference does **exhaustive tree search**:
1. Starts at thumbnail (max_level)
2. For each patch, model decides ZOOM or STOP
3. If ZOOM: **ALL children** are recursively explored
4. If STOP: patch is collected

**Example tree:**
```
Level 3 (patch_0)
├── ZOOM → explore all 4 children
    ├── Level 2 (child_0) → STOP → kept
    ├── Level 2 (child_1) → ZOOM → explore 16 grandchildren
    ├── Level 2 (child_2) → STOP → kept
    └── Level 2 (child_3) → ZOOM → explore 16 grandchildren
```

---

## ⚠️ Semantic Issues Identified

### **Issue 1: Train-Test Exploration Mismatch** ⚠️

**Training:**
- Agent follows **single trajectory** (child 0 only)
- Learns: "At position (x,y,level), should I zoom or stop?"
- Does NOT learn: "Which child should I explore?"

**Inference:**
- Explores **all children** at every zoom
- Assumes model learned: "Is this region worth exploring deeper?"
- But model never learned to choose between multiple paths

**Consequence:**
- Model learns a policy for single-path navigation
- Applied to multi-path exploration during inference
- May work, but not optimal alignment

---

### **Issue 2: Stochastic vs Deterministic Action Selection** ⚠️

**Training:**
```python
action ~ Categorical(logits)  # Samples from distribution
# If logits = [-0.5, 1.2], might sample STOP 27% of time
```

**Inference:**
```python
action = argmax(logits)  # Always picks highest
# If logits = [-0.5, 1.2], ALWAYS picks ZOOM
```

**Consequence:**
- Model trained with exploration (samples actions)
- Inference uses exploitation (greedy actions)
- Standard practice, but combined with Issue 1 creates disconnect

---

### **Issue 3: Reward Signal Mismatch** ⚠️

**Training reward includes:**
```python
reward = InfoGainReward(parent, children)  # Based on all children
        + PatchCost(-0.002)                 # Small penalty per zoom
        + StopPenalty(-0.5)                 # Penalty/reward for stopping
```

**Inference reward:**
- No reward signal at all (greedy policy execution)
- Model must generalize from training distribution

**Consequence:**
- Model optimized for cumulative reward during training
- But at inference, just follows greedy policy
- This is standard RL, but worth noting

---

## 🔧 Potential Problems & Solutions

### **Problem A: Child 0 Bias** 🔴

**Issue:** Model only sees data from child 0 position during training.

**Evidence:** In `env.step()`:
```python
idx = 0  # Always child 0!
```

**Impact:**
- Model may develop positional bias toward top-left quadrant
- May not generalize well to other child positions
- But mitigated by random starting positions during reset()

**Severity:** LOW-MEDIUM
- Random starting positions provide diversity
- But still creates subtle bias

**Solution Options:**
1. **Randomize child selection** during training:
   ```python
   idx = np.random.randint(len(child_patches))
   ```
2. **Use all children** (expensive, creates 4-16x more data):
   ```python
   for idx, child in enumerate(child_patches):
       next_state = encode_state(child_patches[idx])
       # Store transition, but complex episode structure
   ```
3. **Accept current design** - model learns zoom/stop, not path selection

---

### **Problem B: Reward Computed on All Children, But Only One Followed** 🟡

**Issue:** Reward signal includes information from all children, but trajectory only follows one.

**Example:**
```
Parent patch
├── Child 0: tissue (high info gain)
├── Child 1: blank (low info gain)  
├── Child 2: tissue (high info gain)
└── Child 3: artifact (medium info gain)

Reward = InfoGainReward(parent, [ch0, ch1, ch2, ch3])
       = weight * (1.0 - mean([0.8, 0.1, 0.8, 0.4]))
       = weight * 0.475  # Positive, encourages zoom

But model only sees state transition: parent → child_0
```

**Impact:**
- Model gets reward signal aggregated over all children
- But only experiences one child's state
- Creates **credit assignment problem**: which child caused the reward?

**Severity:** MEDIUM
- Model still learns zoom/stop correctly
- But may not learn fine-grained spatial patterns

**Solution:** Already optimal for current task
- Aggregate reward makes sense for "is this region interesting" question
- Individual child selection requires different action space

---

### **Problem C: No "Which Child" Learning** 🟡

**Issue:** Model has no supervision for choosing which child to follow.

**Current Design:**
- Action space: {STOP, ZOOM}
- No action for "zoom to child 0 vs child 1 vs child 2 vs child 3"

**Implication:**
- Training: Doesn't matter, we pick child 0
- Inference: Doesn't matter, we explore all children

**Severity:** LOW
- Current design is internally consistent
- Model learns the right thing for the current task

**Future Enhancement:**
- Extend action space to {STOP, ZOOM_CHILD_0, ZOOM_CHILD_1, ...}
- More complex, but allows learning path selection
- Requires different reward structure

---

## ✅ What IS Working Correctly

### **1. Zoom/Stop Decision Learning** ✅

The model **correctly learns** when to zoom vs stop:
- Reward signal (info gain, entropy, etc.) provides supervision
- High reward for zooming into interesting regions
- Low reward for zooming into blank areas
- Model optimizes this decision effectively

### **2. State Representation** ✅

The 515-dim state provides:
- Spatial context (level, x, y coordinates)
- Visual content (512-dim PLIP embedding)
- Model can learn content-based decisions

### **3. Reward Computation** ✅

Rewards are computed correctly:
- InfoGainReward: measures dissimilarity (correct sign)
- EntropyGainReward: measures information
- PatchCost: penalizes excessive zooming
- StopPenalty: controls zoom/stop trade-off

### **4. Episode Structure** ✅

Training episodes:
- Start at random position (good exploration)
- Terminate naturally (STOP or level 0)
- Collect trajectory for policy update
- Standard A2C/REINFORCE structure

### **5. Inference Logic** ✅

Inference correctly:
- Explores all children when zooming
- Collects patches when stopping
- Handles coordinate scaling properly
- Builds comprehensive patch set

---

## 🎯 Semantic Correctness Assessment

### **Is the training loop semantically correct?** 

**Answer: YES, with caveats** ✅⚠️

**Correct aspects:**
- ✅ Implements valid A2C/REINFORCE algorithm
- ✅ Reward signals match intended behavior
- ✅ Episode structure is sound
- ✅ Gradient updates are correct
- ✅ State representation is appropriate

**Caveats:**
- ⚠️ Single-path training for multi-path inference task
- ⚠️ Child 0 bias (mitigated by random starts)
- ⚠️ Aggregate reward with single-child trajectory

**Conclusion:** The training loop implements a **valid learning algorithm** that will learn a **useful policy** for zoom/stop decisions. The policy may not be **optimal** due to train-test mismatch, but it will **work**.

---

### **Is the inference loop semantically correct?**

**Answer: YES** ✅

**Correct aspects:**
- ✅ Recursive exploration is appropriate for the task
- ✅ Greedy action selection is standard for inference
- ✅ Explores all children (comprehensive coverage)
- ✅ Correctly propagates coordinates
- ✅ Properly categorizes kept vs discarded patches

**No issues found.** Inference loop is semantically correct for the task.

---

### **Do training and inference align semantically?**

**Answer: PARTIAL ALIGNMENT** ⚠️

**What aligns:**
- ✅ Both use same state representation
- ✅ Both use same action space {STOP, ZOOM}
- ✅ Both handle coordinate scaling correctly
- ✅ Inference policy (greedy) is standard for trained policy (stochastic)

**What misaligns:**
- ❌ Training: single-path → Inference: multi-path
- ❌ Training: follows child 0 → Inference: follows all children
- ⚠️ Training: aggregate reward → Inference: no reward

**Impact:** Model will learn a useful policy but may not be optimal due to exploration mismatch.

---

## 💡 Recommendations

### **Option 1: Keep Current Design** [RECOMMENDED]

**Justification:**
- Training is efficient (single-path)
- Inference is comprehensive (multi-path)
- Model learns the core task (zoom vs stop)
- Mismatch is manageable for this application

**Trade-off:**
- Accept suboptimal policy due to train-test gap
- Benefit from faster training and simpler implementation

**Action:** ✅ No changes needed

---

### **Option 2: Match Training to Inference** [EXPENSIVE]

**Change:** Make training explore all children like inference does.

**Implementation:**
```python
# In env.step() for action=1 (ZOOM):
# Instead of picking child 0, create branching episodes
# or use importance sampling to weight all children

# This is complex and computationally expensive
```

**Trade-off:**
- ✅ Better train-test alignment
- ❌ 4-16x more computation per step
- ❌ More complex episode handling
- ❌ May not significantly improve results

**Recommendation:** ❌ Not worth the cost

---

### **Option 3: Randomize Child Selection** [SIMPLE IMPROVEMENT]

**Change:** Randomly pick which child to follow during training.

**Implementation:**
```python
# In dynamic_patch_env.py, line ~220:
idx = np.random.randint(len(child_patches))  # Instead of idx=0
```

**Trade-off:**
- ✅ Reduces child 0 positional bias
- ✅ Simple one-line change
- ✅ No computational overhead
- ✅ Better coverage of state space

**Recommendation:** ✅ **Implement this change**

---

### **Option 4: Extend Action Space** [FUTURE WORK]

**Change:** Add actions for choosing specific children.

**New action space:**
- STOP
- ZOOM_TO_CHILD_0
- ZOOM_TO_CHILD_1
- ZOOM_TO_CHILD_2
- ZOOM_TO_CHILD_3

**Trade-off:**
- ✅ Model learns path selection
- ✅ Better alignment with inference
- ❌ More complex action space
- ❌ Longer training time
- ❌ Requires reward restructuring

**Recommendation:** ⏭️ Consider for future iteration

---

## 📝 Final Verdict

### **Training Loop: SEMANTICALLY CORRECT** ✅

The training loop implements a valid reinforcement learning algorithm (Actor-Critic) that will learn a useful policy for the zoom/stop decision task.

**Minor issues:**
- Child 0 positional bias (fixable with random selection)
- Single-path training for multi-path task (acceptable trade-off)

### **Inference Loop: SEMANTICALLY CORRECT** ✅

The inference loop correctly implements recursive tree exploration with greedy action selection.

**No issues found.**

### **Train-Inference Alignment: PARTIAL** ⚠️

There is a semantic mismatch between single-path training and multi-path inference, but this is an **acceptable design trade-off** for:
- Computational efficiency
- Implementation simplicity
- Task requirements (zoom/stop decision, not path selection)

### **Recommended Action:**

✅ **Implement Option 3: Randomize child selection**

Change line ~220 in `dynamic_patch_env.py`:
```python
# FROM:
idx = 0

# TO:
idx = np.random.randint(len(child_patches))
```

This simple change will:
- Reduce positional bias
- Improve state space coverage
- Better align training with inference
- No computational cost

**Everything else is correct and ready for production use.** ✅

---

**End of Semantic Analysis**
