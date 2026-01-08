# Reward Distinction Testing System

This system helps you visualize and tune reward functions by testing them on diverse sample patches.

## Test
python src/reward_distinction/test_reward_distinction.py --patches src/
reward_distinction/sample_patches

## Quick Start

```bash
./src/reward_distinction/run_distinction_test.sh
```

This will:
1. Extract diverse sample patches from test image
2. Compute rewards for each patch type
3. Generate visualization and report

## Manual Usage

### Step 1: Extract Sample Patches

```bash
python src/reward_distinction/extract_sample_patches.py \
    --image ./data/example_images/test_img_1.svs \
    --output ./src/reward_distinction/sample_patches
```

This extracts patches with different characteristics:
- **blank**: Pure white/background
- **sparse_tissue**: Low tissue density
- **medium_tissue**: Moderate tissue density  
- **dense_tissue**: High tissue density
- **stained**: Heavily stained regions
- **complex**: High visual variance
- **uniform**: Low visual variance

### Step 2: Configure Reward Function

Edit `src/reward_distinction/reward_config.yaml`:

**Option A: Use predefined engine**
```yaml
engine_name: "infogain_only"
```

**Option B: Custom components**
```yaml
components:
  - type: InfoGainReward
    params:
      weight: 3.0
  
  - type: PatchCost
    params:
      cost: 0.3
  
  - type: SimpleStopPenalty
    params:
      penalty: 0.1
```

### Step 3: Test Rewards

```bash
python src/reward_distinction/test_reward_distinction.py \
    --config ./src/reward_distinction/reward_config.yaml \
    --patches ./src/reward_distinction/sample_patches \
    --output ./src/reward_distinction/results
```

## Output

### Visualization (`results/reward_visualization.png`)
- Shows all sample patches
- Color-coded by reward (red=low, green=high)
- Sorted by reward value

### Report (`results/reward_report.txt`)
- Summary statistics (range, mean, std dev)
- Rewards for each patch type
- Quality assessment
- Pattern validation

## Interpreting Results

### Good Distinction
```
Reward range: -0.50 to +0.80
Spread: 1.30
✓✓ EXCELLENT: High spread (>1.0)
```

### Poor Distinction
```
Reward range: -0.05 to +0.08
Spread: 0.13
⚠️ POOR: Very low spread (<0.1)
```

## Tips for Tuning

1. **Check spread**: Want >0.5 for meaningful distinctions
2. **Verify patterns**: Blank patches should score low, dense tissue high
3. **Adjust weights**: Increase InfoGain weight for more distinction
4. **Test penalties**: Make sure stopping is appropriately penalized
5. **Iterate**: Run multiple times with different configs

## Example Workflow

```bash
# Extract patches once
python src/reward_distinction/extract_sample_patches.py

# Try different configs
# Edit reward_config.yaml: weight=2.0, cost=0.3, penalty=0.1
python src/reward_distinction/test_reward_distinction.py
open src/reward_distinction/results/reward_visualization.png

# Edit reward_config.yaml: weight=5.0, cost=0.3, penalty=0.1  
python src/reward_distinction/test_reward_distinction.py
open src/reward_distinction/results/reward_visualization.png

# Compare results to find best config
```

## Advanced: Custom Sample Patches

You can add your own patches to `sample_patches/`:
```bash
cp my_interesting_patch.png src/reward_distinction/sample_patches/custom_1.png
python src/reward_distinction/test_reward_distinction.py
```

## Troubleshooting

**"No samples extracted"**: Image might be mostly blank, try different level
```bash
python src/reward_distinction/extract_sample_patches.py --level 1
```

**"All rewards identical"**: Reward function not sensitive enough
- Increase InfoGain weight
- Reduce cost penalty
- Check embedder is working

**"Unexpected pattern"**: Reward function might have bugs
- Check component parameters
- Verify STOP penalty sign
- Test with simpler config first
