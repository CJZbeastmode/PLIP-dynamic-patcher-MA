Summary
Your observation is spot-on. Here's what's happening:

The Two Failure Modes
No zooming (x) - 11 configs: Agent learns "always STOP" because:

Cost too low (≤0.25) OR weight too high (≥3.75)
InfoGain signal dominates, making surface patches look good enough
Infinite zooming (err) - 20 configs: Agent learns "always ZOOM" because:

Cost too high (≥0.35) OR penalty too strong (≤-0.25)
Stopping becomes too expensive, agent zooms to timeout
Why Same Config Gives Different Results?
YES - because your data is too limited:

Your dataset: 60 images × 5 episodes = 300 training episodes
Your model: 131,000 parameters
Ratio: 0.23 samples per parameter (need ~10+)
You're underfit by 40×! This causes:

High variance in gradient updates
Random initialization matters a lot
Different episodic order → different convergence
Some runs get lucky, some don't
The "Goldilocks Zone" That Works
Only ~24% of configs succeeded. They cluster around:

Weight: 2.25-3.0
Cost: 0.3-0.4
Penalty: -0.2 to -0.3
Rule of thumb discovered:

cost ≈ 0.1 × weight
penalty ≈ -0.07 × weight
What To Do Next
Priority 1: More data

EPISODES_PER_WSI = 10   # was 5
Priority 2: Focus search on proven zone

Only test configs with w=2.0-3.0, c=0.25-0.4, p=-0.15 to -0.3
Discard extreme values that always fail
I've created INFERENCE_ANALYSIS.md with complete analysis, mathematical explanations, and 7 specific recommendations to fix the training instability.
