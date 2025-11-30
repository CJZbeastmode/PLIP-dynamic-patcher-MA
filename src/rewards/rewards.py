# rewards.py
#
# Clean, structured RewardEngine configurations.
# Minimal (1) and DepthPenalty (2) sections are preserved EXACTLY as provided.

from .reward_module import (
    # Vision-based zoom rewards
    InfoGainReward,
    MaxDifferenceInfoReward,
    MaxDifferenceRangeInfoReward,
    CosineSimilarityReward,

    # Entropy-based zoom rewards
    EntropyGainReward,
    MaxEntropyGainReward,

    # Extras
    TissuePresenceReward,
    TextAlignReward,
    SemanticSurpriseReward,
    TextAlignFixedEmbeddingReward,

    # Penalties
    PatchCost,
    StopPenalty,
    DepthPenalty,
    ZoomBudgetPenalty,
    InvalidPatchPenalty,

    # Engine
    RewardEngine,
    EmbeddingComputer,
)

from transformers import CLIPModel, CLIPProcessor

# ========================================================
# Load PLIP model + processor
# ========================================================
plip_model = CLIPModel.from_pretrained("vinid/plip")
plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
plip_model.eval()

embedder = EmbeddingComputer(
    model=plip_model,
    processor=plip_processor,
    faiss=None,
    text_embed=None
)


# ============================================================
# SECTION 1 — Minimal Engines
# ============================================================

infogain_only = RewardEngine([
    InfoGainReward(weight=10, embedder=embedder),
    PatchCost(cost=0.001),
    StopPenalty(penalty=0.25, embedder=embedder),
])

info_gain_max_difference_only = RewardEngine([
    MaxDifferenceInfoReward(weight=10, embedder=embedder),
    PatchCost(cost=0.001),
    StopPenalty(penalty=0.25, embedder=embedder),
])

info_gain_max_range_only = RewardEngine([
    MaxDifferenceRangeInfoReward(weight=10, embedder=embedder),
    PatchCost(cost=0.001),
    StopPenalty(penalty=0.25, embedder=embedder),
])

cos_sim_only = RewardEngine([
    CosineSimilarityReward(weight=10, embedder=embedder),
    PatchCost(cost=0.001),
    StopPenalty(penalty=0.25, embedder=embedder),
])

entropy_only = RewardEngine([
    EntropyGainReward(weight=8),
    PatchCost(cost=0.001),
    StopPenalty(penalty=0.25, embedder=embedder),
])

max_entropy_only = RewardEngine([
    MaxEntropyGainReward(weight=10),
    PatchCost(cost=0.001),
    StopPenalty(penalty=0.25, embedder=embedder),
])

text_align_only = RewardEngine([
    TextAlignReward(weight=12, embedder=embedder),
    PatchCost(cost=0.001),
    StopPenalty(penalty=0.25, embedder=embedder),
])

text_align_fixed_embedding_only = RewardEngine([
    TextAlignFixedEmbeddingReward(weight=12, embedder=embedder),
    PatchCost(cost=0.001),
    StopPenalty(penalty=0.25, embedder=embedder),
])


# ============================================================
# SECTION 2 — Minimal + DepthPenalty
# ============================================================

infogain_only_depth = RewardEngine([
    InfoGainReward(weight=10, embedder=embedder),
    PatchCost(cost=0.001),
    DepthPenalty(weight=0.005),
    StopPenalty(penalty=0.25, embedder=embedder),
])

info_gain_max_difference_only_depth = RewardEngine([
    MaxDifferenceInfoReward(weight=10, embedder=embedder),
    PatchCost(cost=0.001),
    DepthPenalty(weight=0.005),
    StopPenalty(penalty=0.25, embedder=embedder),
])

info_gain_max_range_only_depth = RewardEngine([
    MaxDifferenceRangeInfoReward(weight=10, embedder=embedder),
    PatchCost(cost=0.001),
    DepthPenalty(weight=0.005),
    StopPenalty(penalty=0.25, embedder=embedder),
])

cos_sim_only_depth = RewardEngine([
    CosineSimilarityReward(weight=10, embedder=embedder),
    PatchCost(cost=0.001),
    DepthPenalty(weight=0.005),
    StopPenalty(penalty=0.25, embedder=embedder),
])

entropy_only_depth = RewardEngine([
    EntropyGainReward(weight=8),
    PatchCost(cost=0.001),
    DepthPenalty(weight=0.005),
    StopPenalty(penalty=0.25, embedder=embedder),
])

max_entropy_only_depth = RewardEngine([
    MaxEntropyGainReward(weight=10),
    PatchCost(cost=0.001),
    DepthPenalty(weight=0.005),
    StopPenalty(penalty=0.25, embedder=embedder),
])

text_align_only_depth = RewardEngine([
    TextAlignReward(weight=12, embedder=embedder),
    PatchCost(cost=0.001),
    DepthPenalty(weight=0.005),
    StopPenalty(penalty=0.25, embedder=embedder),
])

text_align_fixed_embedding_only_depth = RewardEngine([
    TextAlignFixedEmbeddingReward(weight=12, embedder=embedder),
    PatchCost(cost=0.001),
    DepthPenalty(weight=0.005),
    StopPenalty(penalty=0.25, embedder=embedder),
])


# ============================================================
# SECTION 3 — Ablation Engines (new, systematic)
# ============================================================

ablation_infogain_only = RewardEngine([
    InfoGainReward(weight=10, embedder=embedder),
])

ablation_max_difference_only = RewardEngine([
    MaxDifferenceInfoReward(weight=12, embedder=embedder),
])

ablation_cosine_range_only = RewardEngine([
    MaxDifferenceRangeInfoReward(weight=12, embedder=embedder),
])

ablation_entropy_only = RewardEngine([
    EntropyGainReward(weight=10),
])

ablation_max_entropy_only = RewardEngine([
    MaxEntropyGainReward(weight=10),
])

ablation_text_align_only = RewardEngine([
    TextAlignReward(weight=12, embedder=embedder),
])

ablation_no_vision = RewardEngine([
    TextAlignReward(weight=12, embedder=embedder),
    SemanticSurpriseReward(weight=3, embedder=embedder),
])

ablation_no_multimodal = RewardEngine([
    InfoGainReward(weight=12, embedder=embedder),
    EntropyGainReward(weight=3),
])


# ============================================================
# SECTION 4 — Vision-Dominant Engines
# ============================================================

vision_contrast = RewardEngine([
    MaxDifferenceRangeInfoReward(weight=10, embedder=embedder),
    EntropyGainReward(weight=3),
    TissuePresenceReward(weight=2),
    PatchCost(cost=0.001),
])

vision_structural = RewardEngine([
    MaxEntropyGainReward(weight=8),
    TissuePresenceReward(weight=3),
    PatchCost(cost=0.002),
    DepthPenalty(weight=0.01),
])


# ============================================================
# SECTION 5 — Multimodal-Focused Engines
# ============================================================

multimodal_text_driven = RewardEngine([
    TextAlignReward(weight=12, embedder=embedder),
    SemanticSurpriseReward(weight=4, embedder=embedder),
    PatchCost(cost=0.002),
    StopPenalty(penalty=0.2, embedder=embedder),
])

multimodal_entropy = RewardEngine([
    MaxEntropyGainReward(weight=5),
    TextAlignReward(weight=10, embedder=embedder),
    TissuePresenceReward(weight=2),
    PatchCost(cost=0.002),
])

multimodal_with_vision = RewardEngine([
    InfoGainReward(weight=6, embedder=embedder),
    TextAlignReward(weight=8, embedder=embedder),
    SemanticSurpriseReward(weight=3, embedder=embedder),
    TissuePresenceReward(weight=1),
    PatchCost(cost=0.002),
])


# ============================================================
# SECTION 6 — Balanced / Hybrid Engines
# ============================================================

balanced_standard = RewardEngine([
    InfoGainReward(weight=7, embedder=embedder),
    TextAlignReward(weight=5, embedder=embedder),
    TissuePresenceReward(weight=2),
    PatchCost(cost=0.002),
    StopPenalty(penalty=0.25, embedder=embedder),
    DepthPenalty(weight=0.01),
    InvalidPatchPenalty(penalty=1),
])

balanced_comprehensive = RewardEngine([
    InfoGainReward(weight=6, embedder=embedder),
    TextAlignReward(weight=7, embedder=embedder),
    SemanticSurpriseReward(weight=2, embedder=embedder),
    EntropyGainReward(weight=2),
    TissuePresenceReward(weight=1.5),
    PatchCost(cost=0.002),
    StopPenalty(penalty=0.25, embedder=embedder),
    DepthPenalty(weight=0.01),
    ZoomBudgetPenalty(weight=0.5, budget=4),
    InvalidPatchPenalty(penalty=1),
])

balanced_diversity = RewardEngine([
    MaxDifferenceRangeInfoReward(weight=8, embedder=embedder),
    MaxEntropyGainReward(weight=4),
    TextAlignReward(weight=6, embedder=embedder),
    TissuePresenceReward(weight=2),
    PatchCost(cost=0.002),
    StopPenalty(penalty=0.25, embedder=embedder),
])


# ============================================================
# SECTION 7 — Special-Purpose Engines
# ============================================================

depth_limited = RewardEngine([
    InfoGainReward(weight=8, embedder=embedder),
    TissuePresenceReward(weight=2),
    PatchCost(cost=0.002),
    DepthPenalty(weight=0.05),
    ZoomBudgetPenalty(weight=2.0, budget=2),
])

fast_exploration = RewardEngine([
    MaxDifferenceInfoReward(weight=10, embedder=embedder),
    PatchCost(cost=0.0001),
])

quality_assurance = RewardEngine([
    InfoGainReward(weight=5, embedder=embedder),
    TissuePresenceReward(weight=5, blank_thr=220),
    InvalidPatchPenalty(penalty=3.0),
    PatchCost(cost=0.003),
])


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

    # Ablation
    "ablation_infogain_only": ablation_infogain_only,
    "ablation_max_difference_only": ablation_max_difference_only,
    "ablation_cosine_range_only": ablation_cosine_range_only,
    "ablation_entropy_only": ablation_entropy_only,
    "ablation_max_entropy_only": ablation_max_entropy_only,
    "ablation_text_align_only": ablation_text_align_only,
    "ablation_no_vision": ablation_no_vision,
    "ablation_no_multimodal": ablation_no_multimodal,

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
}
