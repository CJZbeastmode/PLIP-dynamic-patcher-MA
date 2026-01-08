from torch.nn.functional import cosine_similarity
import numpy as np
import cv2


class PatchScoreModule:
    """
    Abstract base class for all patch scoring modules.

    This interface defines the minimal contract required by the
    DynamicPatchEnv and downstream RL agents.

    Design assumptions:
    -------------------
    - All methods must be side-effect free.
    - No method may raise exceptions during training or inference.
    - Any failure must degrade gracefully to a neutral score (typically 0.0).
    - Scores are assumed to be comparable within a module, but not necessarily
      across different modules.

    Each concrete subclass encodes a different inductive bias about what
    constitutes a “good” zoom decision in a WSI.
    """

    def compute_stop(self, **kwargs):
        """Score associated with terminating exploration at the current patch."""
        raise NotImplementedError

    def compute_zoom(self, **kwargs):
        """Score associated with zooming into child patches."""
        raise NotImplementedError

    def compute_diff(self, **kwargs):
        """
        Convenience method computing the relative preference for ZOOM vs STOP.
        Typically implemented as: s_zoom − s_stop.
        """
        raise NotImplementedError

    def infer(self, **kwargs):
        """
        Deterministic inference rule mapping scores to an action.
        Used for greedy or evaluation-time policies.
        """
        raise NotImplementedError

    def rl_parameters(self, **kwargs):
        """
        Optional module-specific RL hyperparameters.
        These values may be used to tune exploration, penalties, or learning rate.
        """
        return NotImplementedError


# ==========================================================
# Image similarity score
# ==========================================================
class ImgSimScore(PatchScoreModule):
    """
    Image-similarity–based score.

    Motivation:
    -----------
    Zooming is rewarded when child patches are *visually dissimilar*
    from the parent patch, encouraging exploration of new visual content.

    This score is agnostic to semantics and relies purely on embedding
    geometry, making it a strong unsupervised baseline.
    """

    def __init__(self, weight=10.0, embedder=None, agg="mean", **kwargs):
        if agg not in ["mean", "max"]:
            raise ValueError(f"Invalid aggregation method: {agg}")

        self.weight = weight  # Global scaling factor for reward magnitude
        self.embedder = embedder  # Shared image embedding backend (PLIP / CONCH)
        self.agg = agg  # Aggregation method for multiple children

    def compute_stop(self, **kwargs):
        """
        STOP has no intrinsic reward in this module.
        All preference is expressed via the zoom score.
        """
        return 1.0  # Only meaningful structural difference would result in zooming

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Compute zoom reward as inverse similarity between parent and children.

        Similarity is measured via cosine similarity in embedding space.
        The reward is proportional to (1 − similarity), i.e. information gain.
        """
        if parent_patch is None or not child_patches or self.embedder is None:
            return 0.0

        try:
            ep = self.embedder.img_emb(parent_patch)
        except Exception:
            return 0.0

        sims = []
        for p in child_patches:
            try:
                ec = self.embedder.img_emb(p)
                sim_t = cosine_similarity(ep, ec, dim=0)
                sim_val = (
                    float(sim_t.mean().item())
                    if sim_t.numel() > 1
                    else float(sim_t.item())
                )
                sims.append(sim_val)
            except Exception:
                continue

        if len(sims) == 0:
            return 0.0

        try:
            if self.agg == "mean":
                return self.weight * (1.0 - float(np.mean(sims)))
            elif self.agg == "max":
                # Conservative variant: penalize the most similar child
                return self.weight * (1.0 - float(np.min(sims)))
        except Exception:
            return 0.0

        return 0.0

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Relative preference for zooming vs stopping.
        """
        try:
            s_stop = self.compute_stop(parent_patch=parent_patch, **kwargs)
            s_zoom = self.compute_zoom(
                parent_patch=parent_patch,
                child_patches=child_patches,
                agg=self.agg,
                **kwargs,
            )
            return s_zoom - s_stop
        except Exception:
            return 0.0

    def infer(self, s_stop, s_zoom):
        """
        Greedy inference rule:
        - If both scores are non-positive, STOP.
        - Otherwise choose the higher score.
        """
        try:
            if s_stop <= 0 and s_zoom <= 0:
                return 0
            return 1 if (s_zoom >= s_stop) else 0
        except Exception:
            return 0

    def rl_parameters(self):
        """
        Recommended RL hyperparameters for this reward structure.

        The values reflect:
        - Dense, smooth rewards
        - Short-horizon exploration
        - Mild entropy regularization
        """
        return {
            "ZOOM_COST": 0.3,
            "DEPTH_COST": 0.05,
            "MAX_ZOOM_FRAC": 0.5,
            "OVERZOOM_PENALTY": 0.0,
            "ENTROPY_BETA": 0.02,
            "GAMMA": 0.95,
            "LR": 1e-4,
        }


# ==========================================================
# Text–image alignment score
# ==========================================================
class TextAlignScore(PatchScoreModule):
    """
    Semantic alignment score between image patches and pathology text prompts.

    Motivation:
    -----------
    Zooming is encouraged when child patches increase semantic alignment
    with domain-specific pathology concepts.

    This module introduces weak supervision via language.
    """

    def __init__(self, weight=1.0, embedder=None, k=3, agg="mean", **kwargs):
        if agg not in ["mean", "max"]:
            raise ValueError(f"Invalid aggregation method: {agg}")

        self.weight = weight
        self.embedder = embedder  # Shared vision–language model
        self.k = k  # Top-k aggregation (currently implicit)
        self.agg = agg

    def compute_stop(self, parent_patch=None, **kwargs):
        """
        STOP score reflects how well the current patch aligns with pathology text.
        """
        if parent_patch is None or self.embedder is None:
            return 0.0
        try:
            s = self.embedder.text_sim(parent_patch)
            return self.weight * float(s)
        except Exception:
            return 0.0

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Zoom score aggregates semantic alignment across child patches.
        """
        if not child_patches or self.embedder is None:
            return 0.0

        scores = []
        for p in child_patches:
            try:
                scores.append(self.embedder.text_sim(p, aggregate=self.agg))
            except Exception:
                continue

        if len(scores) == 0:
            return 0.0

        try:
            if self.agg == "mean":
                return self.weight * float(np.mean(scores))
            elif self.agg == "max":
                return self.weight * float(np.max(scores))
        except Exception:
            return 0.0

        return 0.0

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Relative semantic gain from zooming.
        """
        try:
            s_stop = self.compute_stop(parent_patch=parent_patch, **kwargs)
            s_zoom = self.compute_zoom(
                parent_patch=parent_patch,
                child_patches=child_patches,
                agg=self.agg,
                **kwargs,
            )
            return s_zoom - s_stop
        except Exception:
            return 0.0

    def infer(self, s_stop, s_zoom):
        """Greedy semantic decision rule."""
        try:
            return 1 if (s_zoom >= s_stop) else 0
        except Exception:
            return 0

    def rl_parameters(self):
        """
        RL hyperparameters tuned for sparse, higher-variance semantic rewards.
        """
        return {
            "ZOOM_COST": 2.0,
            "DEPTH_COST": 0.1,
            "MAX_ZOOM_FRAC": 0.5,
            "OVERZOOM_PENALTY": 0.0,
            "ENTROPY_BETA": 0.01,
            "GAMMA": 0.95,
            "LR": 1e-4,
        }


# ==========================================================
# Tissue presence (reward-only)
# ==========================================================
class TissuePresenceScore(PatchScoreModule):
    """
    Binary tissue presence reward.

    Motivation:
    -----------
    Encourages exploration of regions containing tissue while remaining
    neutral toward blank/background regions.

    This module provides a very simple heuristic baseline.
    """

    def __init__(self, weight=1.0, blank_thr=230, agg="any", **kwargs):
        if agg not in ["any", "all"]:
            raise ValueError(f"Invalid aggregation method: {agg}")

        self.weight = weight
        self.blank_thr = blank_thr  # Mean intensity threshold for blank detection
        self.agg = agg

    def _is_blank(self, patch):
        """Heuristic blank detection via mean pixel intensity."""
        try:
            return np.array(patch).mean() > self.blank_thr
        except Exception:
            return True

    def compute_stop(self, parent_patch=None, **kwargs):
        """Reward stopping if the current patch contains tissue."""
        if parent_patch is None:
            return 0.0
        try:
            return self.weight if not self._is_blank(parent_patch) else 0.0
        except Exception:
            return 0.0

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        """
        Reward zooming if tissue is detected among child patches.
        """
        if not child_patches:
            return 0.0

        try:
            if self.agg == "any":
                return (
                    self.weight
                    if any(not self._is_blank(p) for p in child_patches)
                    else 0.0
                )
            elif self.agg == "all":
                return (
                    self.weight
                    if all(not self._is_blank(p) for p in child_patches)
                    else 0.0
                )
        except Exception:
            return 0.0

        return 0.0

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        """Relative tissue gain from zooming."""
        try:
            return self.compute_zoom(
                parent_patch=parent_patch,
                child_patches=child_patches,
                agg=self.agg,
                **kwargs,
            ) - self.compute_stop(parent_patch=parent_patch, **kwargs)
        except Exception:
            return 0.0

    def infer(self, s_stop, s_zoom):
        """Prefer zoom if it increases tissue presence."""
        try:
            return 1 if (s_zoom > s_stop) else 0
        except Exception:
            return 0


# ==========================================================
# Tissue presence (explicit penalty version)
# ==========================================================
class TissuePresencePenalty(PatchScoreModule):
    """
    Signed tissue reward with explicit penalties.

    Motivation:
    -----------
    Explicitly penalizes blank/background regions instead of merely
    withholding reward. This produces sharper gradients and stronger
    exploration pressure.

    This module is *not* equivalent to TissuePresenceScore and should
    be treated as a distinct experimental condition.
    """

    def __init__(self, weight=1.0, blank_thr=230, agg="any", **kwargs):
        if agg not in ["any", "all"]:
            raise ValueError(f"Invalid aggregation method: {agg}")
        self.weight = weight
        self.blank_thr = blank_thr
        self.agg = agg

    def _is_blank(self, patch):
        try:
            return np.array(patch).mean() > self.blank_thr
        except Exception:
            return True

    def compute_stop(self, parent_patch=None, **kwargs):
        """Reward tissue, penalize blank at the current level."""
        if parent_patch is None:
            return 0.0
        try:
            return self.weight if not self._is_blank(parent_patch) else -self.weight
        except Exception:
            return 0.0

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        """Signed reward based on tissue presence in child patches."""
        if not child_patches:
            return 0.0

        try:
            if self.agg == "any":
                return (
                    self.weight
                    if any(not self._is_blank(p) for p in child_patches)
                    else -self.weight
                )
            elif self.agg == "all":
                return (
                    self.weight
                    if all(not self._is_blank(p) for p in child_patches)
                    else -self.weight
                )
        except Exception:
            return 0.0

        return 0.0

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        """Relative signed advantage of zooming."""
        try:
            return self.compute_zoom(
                parent_patch=parent_patch,
                child_patches=child_patches,
                agg=self.agg,
                **kwargs,
            ) - self.compute_stop(parent_patch=parent_patch, **kwargs)
        except Exception:
            return 0.0

    def infer(self, s_stop, s_zoom):
        """Zoom if it reduces blankness relative to stopping."""
        try:
            return 1 if (s_zoom > s_stop) else 0
        except Exception:
            return 0

    def rl_parameters(self):
        """
        Hyperparameters reflecting discrete, high-contrast rewards.
        """
        return {
            "ZOOM_COST": 0.2,
            "DEPTH_COST": 0.05,
            "MAX_ZOOM_FRAC": 0.5,
            "OVERZOOM_PENALTY": 0.0,
            "ENTROPY_BETA": 0.03,
            "GAMMA": 0.9,
            "LR": 1e-4,
        }


# ==========================================================
# Entropy-based score
# ==========================================================
class EntropyScore(PatchScoreModule):
    """
    Information-theoretic score based on relative grayscale entropy gain.

    Zooming is encouraged only if child patches exhibit a
    meaningful relative increase in entropy compared to the parent.
    """

    def __init__(self, weight=1.0, agg="max", tau=0.01, **kwargs):
        if agg not in ["mean", "max"]:
            raise ValueError(f"Invalid aggregation method: {agg}")
        self.weight = weight
        self.agg = agg
        self.tau = tau  # relative entropy gain threshold

    def _entropy(self, img_np):
        """Compute Shannon entropy of grayscale intensity distribution."""
        if img_np is None:
            return 0.0

        try:
            if img_np.ndim == 2:
                gray = img_np
            else:
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_sum = hist.sum()
            if hist_sum <= 0:
                return 0.0

            hist = hist / hist_sum
            hist = np.maximum(hist, 1e-12)
            return float(-(hist * np.log(hist)).sum())
        except Exception:
            return 0.0

    def compute_stop(self, parent_patch=None, **kwargs):
        if parent_patch is None:
            return 0.0
        return self.weight * self._entropy(np.array(parent_patch))

    def compute_zoom(self, parent_patch=None, child_patches=None, **kwargs):
        if not child_patches:
            return 0.0

        entropies = []
        for cp in child_patches:
            e = self._entropy(np.array(cp))
            if e > 0:
                entropies.append(e)

        if len(entropies) == 0:
            return 0.0

        if self.agg == "mean":
            return self.weight * float(np.mean(entropies))
        elif self.agg == "max":
            return self.weight * float(np.max(entropies))

        return 0.0

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        s_stop = self.compute_stop(parent_patch=parent_patch)
        if s_stop <= 0:
            return 0.0

        s_zoom = self.compute_zoom(
            parent_patch=parent_patch,
            child_patches=child_patches,
        )

        # Relative entropy gain
        return (s_zoom - s_stop) / (s_stop + 1e-6)

    def infer(self, s_stop, s_zoom):
        if s_stop <= 0:
            return 0
        diff = (s_zoom - s_stop) / (s_stop + 1e-6)
        return 1 if diff >= self.tau else 0


# ==========================================================
# Registry
# ==========================================================
PATCH_SCORE_MODULES = {
    "img_sim_score": ImgSimScore,
    "text_align_score": TextAlignScore,
    "tissue_presence_score": TissuePresenceScore,
    "tissue_presence_penalty": TissuePresencePenalty,
    "entropy_score": EntropyScore,
}
