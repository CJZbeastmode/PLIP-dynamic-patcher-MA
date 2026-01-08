# rewards.py
#
# Clean, structured RewardEngine configurations.
# Minimal (1) and DepthPenalty (2) sections are preserved EXACTLY as provided.

from .reward_module import *


# ============================================================
# SECTION 1 — Minimal Engines
# ============================================================

infogain_only = RewardEngine(
    [
        InfoGainReward(weight=5.0),
        PatchCost(cost=0.8),
        SimpleStopPenalty(penalty=0.1),
        TissuePresenceReward(weight=2.0, blank_thr=200),
        InvalidPatchPenalty(penalty=1.0),
    ]
)

info_gain_max_difference_only = RewardEngine(
    [
        MaxDifferenceInfoGainReward(weight=3.0),
        PatchCost(cost=0.55),
        SimpleStopPenalty(penalty=-0.15),
    ]
)

info_gain_max_range_only = RewardEngine(
    [
        MaxDifferenceRangeInfoReward(weight=3.0),
        PatchCost(cost=0.55),
        SimpleStopPenalty(penalty=-0.15),
    ]
)

cos_sim_only = RewardEngine(
    [
        CosineSimilarityReward(weight=3.0),
        PatchCost(cost=0.55),
        SimpleStopPenalty(penalty=-0.15),
    ]
)

entropy_only = RewardEngine(
    [
        EntropyGainReward(weight=3.0),
        PatchCost(cost=0.55),
        SimpleStopPenalty(penalty=-0.15),
    ]
)

max_entropy_only = RewardEngine(
    [
        MaxEntropyGainReward(weight=3.0),
        PatchCost(cost=0.55),
        SimpleStopPenalty(penalty=-0.15),
    ]
)

text_align_only = RewardEngine([TextAlignReward(weight=100.0)])

text_align_fixed_embedding_only = RewardEngine(
    [
        # TextAlignFixedEmbeddingReward(weight=100.0),
        PatchCost(cost=0.55),
        SimpleStopPenalty(penalty=-0.15),
    ]
)


# ============================================================
# SECTION 2 — Minimal + DepthPenalty
# ============================================================

infogain_only_depth = RewardEngine(
    [
        InfoGainReward(weight=3.0),
        PatchCost(cost=0.005),
        SimpleStopPenalty(penalty=0.3),
        DepthPenalty(weight=0.01),
    ]
)

info_gain_max_difference_only_depth = RewardEngine(
    [
        MaxDifferenceInfoGainReward(weight=3),
        PatchCost(cost=0.005),
        DepthPenalty(weight=0.003),
        SimpleStopPenalty(penalty=0.25),
    ]
)

info_gain_max_range_only_depth = RewardEngine(
    [
        MaxDifferenceRangeInfoReward(weight=3),
        PatchCost(cost=0.005),
        DepthPenalty(weight=0.003),
        SimpleStopPenalty(penalty=0.25),
    ]
)

cos_sim_only_depth = RewardEngine(
    [
        CosineSimilarityReward(weight=3),
        PatchCost(cost=0.005),
        DepthPenalty(weight=0.003),
        SimpleStopPenalty(penalty=0.25),
    ]
)

entropy_only_depth = RewardEngine(
    [
        EntropyGainReward(weight=3),
        PatchCost(cost=0.005),
        DepthPenalty(weight=0.003),
        SimpleStopPenalty(penalty=0.25),
    ]
)

max_entropy_only_depth = RewardEngine(
    [
        MaxEntropyGainReward(weight=3),
        PatchCost(cost=0.005),
        DepthPenalty(weight=0.003),
        SimpleStopPenalty(penalty=0.25),
    ]
)

text_align_only_depth = RewardEngine(
    [
        TextAlignReward(weight=6),
        PatchCost(cost=0.005),
        DepthPenalty(weight=0.003),
        SimpleStopPenalty(penalty=0.25),
    ]
)

text_align_fixed_embedding_only_depth = RewardEngine(
    [
        # TextAlignFixedEmbeddingReward(weight=6),
        PatchCost(cost=0.005),
        DepthPenalty(weight=0.003),
        SimpleStopPenalty(penalty=0.25),
    ]
)


# ============================================================
# SECTION 4 — Vision-Dominant Engines
# ============================================================

vision_contrast = RewardEngine(
    [
        MaxDifferenceRangeInfoReward(weight=4),
        EntropyGainReward(weight=2),
        TissuePresenceReward(weight=1.5),
        PatchCost(cost=0.005),
    ]
)

vision_structural = RewardEngine(
    [
        MaxEntropyGainReward(weight=4),
        TissuePresenceReward(weight=2),
        PatchCost(cost=0.007),
        DepthPenalty(weight=0.003),
    ]
)


# ============================================================
# SECTION 5 — Multimodal-Focused Engines
# ============================================================

multimodal_text_driven = RewardEngine(
    [
        TextAlignReward(weight=7),
        SemanticSurpriseReward(weight=2),
        PatchCost(cost=0.005),
        SimpleStopPenalty(penalty=0.25),
    ]
)

multimodal_entropy = RewardEngine(
    [
        MaxEntropyGainReward(weight=3),
        TextAlignReward(weight=6),
        TissuePresenceReward(weight=1.5),
        PatchCost(cost=0.005),
    ]
)

multimodal_with_vision = RewardEngine(
    [
        InfoGainReward(weight=3.0),
        TextAlignReward(weight=5),
        SemanticSurpriseReward(weight=1.5),
        TissuePresenceReward(weight=1),
        PatchCost(cost=0.005),
    ]
)


# ============================================================
# SECTION 6 — Balanced / Hybrid Engines
# ============================================================

balanced_standard = RewardEngine(
    [
        InfoGainReward(weight=3.0),
        TextAlignReward(weight=3),
        TissuePresenceReward(weight=1.5),
        PatchCost(cost=0.005),
        SimpleStopPenalty(penalty=0.25),
        DepthPenalty(weight=0.003),
        InvalidPatchPenalty(penalty=1),
    ]
)

balanced_comprehensive = RewardEngine(
    [
        InfoGainReward(weight=3.0),
        TextAlignReward(weight=4),
        SemanticSurpriseReward(weight=1.5),
        EntropyGainReward(weight=1.5),
        TissuePresenceReward(weight=1),
        PatchCost(cost=0.005),
        SimpleStopPenalty(penalty=0.25),
        DepthPenalty(weight=0.003),
        ZoomBudgetPenalty(weight=0.3, budget=4),
        InvalidPatchPenalty(penalty=1),
    ]
)

balanced_diversity = RewardEngine(
    [
        MaxDifferenceRangeInfoReward(weight=4),
        MaxEntropyGainReward(weight=2),
        TextAlignReward(weight=4),
        TissuePresenceReward(weight=1.5),
        PatchCost(cost=0.005),
        SimpleStopPenalty(penalty=0.25),
    ]
)


# ============================================================
# SECTION 7 — Special-Purpose Engines
# ============================================================

depth_limited = RewardEngine(
    [
        InfoGainReward(weight=3.0),
        TissuePresenceReward(weight=2),
        PatchCost(cost=0.002),
        DepthPenalty(weight=0.05),
        ZoomBudgetPenalty(weight=2.0, budget=2),
    ]
)

fast_exploration = RewardEngine(
    [
        MaxDifferenceInfoGainReward(weight=10),
        PatchCost(cost=0.02),  # Increased from 0.0001
    ]
)

quality_assurance = RewardEngine(
    [
        InfoGainReward(weight=3.0),
        TissuePresenceReward(weight=5, blank_thr=220),
        InvalidPatchPenalty(penalty=3.0),
        PatchCost(cost=0.003),
    ]
)


# ============================================================
# Custom / User-defined Engines
# ============================================================

pure_infogain_engine_supervised = RewardEngine(
    [
        # Use MaxDifferenceInfoReward to emphasize large parent-child dissimilarity
        MaxDifferenceInfoGainReward(weight=10.0),
        # Penalize blank patches (structural penalty)
        TissuePresencePenalty(weight=0.5, blank_thr=230),
        # Zoom cost
        PatchCost(cost=0.5),
        # Simple stop penalty (negative value here will act as a small reward for stopping)
        SimpleStopPenalty(penalty=-0.5),
    ]
)

tissue_presence_penalty_only = RewardEngine(
    [
        TissuePresenceReward(weight=1.0, blank_thr=240),
    ]
)


# ============================================================
# ENGINE CATALOG
# ============================================================

ENGINES = {
    # Minimal
    "infogain_only": infogain_only,
    "info_gain_max_difference_only": info_gain_max_difference_only,
    "info_gain_max_range_only": info_gain_max_range_only,
    "cos_sim_only": cos_sim_only,
    "entropy_only": entropy_only,
    "max_entropy_only": max_entropy_only,
    "text_align_only": text_align_only,
    "text_align_fixed_embedding_only": text_align_fixed_embedding_only,
    # Minimal + Depth
    "infogain_only_depth": infogain_only_depth,
    "info_gain_max_difference_only_depth": info_gain_max_difference_only_depth,
    "info_gain_max_range_only_depth": info_gain_max_range_only_depth,
    "cos_sim_only_depth": cos_sim_only_depth,
    "entropy_only_depth": entropy_only_depth,
    "max_entropy_only_depth": max_entropy_only_depth,
    "text_align_only_depth": text_align_only_depth,
    "text_align_fixed_embedding_only_depth": text_align_fixed_embedding_only_depth,
    # Vision
    "vision_contrast": vision_contrast,
    "vision_structural": vision_structural,
    # Multimodal
    "multimodal_text_driven": multimodal_text_driven,
    "multimodal_entropy": multimodal_entropy,
    "multimodal_with_vision": multimodal_with_vision,
    # Balanced
    "balanced_standard": balanced_standard,
    "balanced_comprehensive": balanced_comprehensive,
    "balanced_diversity": balanced_diversity,
    # Special
    "depth_limited": depth_limited,
    "fast_exploration": fast_exploration,
    "quality_assurance": quality_assurance,
    # Real
    "pure_infogain_engine_supervised": pure_infogain_engine_supervised,
    "tissue_presence_penalty_only": tissue_presence_penalty_only,
}
