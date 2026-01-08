import numpy as np
import torch
import cv2
from torch.nn.functional import cosine_similarity
import torch
from src.utils.embedder import Embedder

# ============================================================
#  Base Class
# ============================================================


class RewardModule:
    def compute(self, **kwargs):
        raise NotImplementedError


class PatchScoreModule:
    def compute_stop():
        raise NotImplementedError

    def compute_zoom():
        raise NotImplementedError

    def compute_diff():
        raise NotImplementedError

    def inference(s_stop, s_zoom):
        raise NotImplementedError


class ImgSimScore(PatchScoreModule):
    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute_stop(self, **kwargs):
        return 0.0

    def compute_zoom(self, parent_patch=None, child_patches=None, agg="mean", **kwargs):
        if parent_patch is None or not child_patches or len(child_patches) == 0:
            return 0.0
        ep = self.embedder.img_emb(parent_patch)

        sims = []
        for p in child_patches:
            ec = self.embedder.img_emb(p)
            sim_t = cosine_similarity(ep, ec, dim=0)
            sim_val = (
                float(sim_t.mean().item()) if sim_t.numel() > 1 else float(sim_t.item())
            )
            sims.append(sim_val)

        # Information gain: reward dissimilarity (1 - similarity)
        if agg == "mean":
            return self.weight * (1.0 - np.mean(sims))
        elif agg == "max":
            return self.weight * (1.0 - np.min(sims))
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

    def compute_diff(self, parent_patch=None, child_patches=None, agg="mean", **kwargs):
        if parent_patch is None or not child_patches or len(child_patches) == 0:
            return 0.0

        s_stop = self.compute_stop(parent_patch=parent_patch, **kwargs)
        s_zoom = self.compute_zoom(
            parent_patch=parent_patch, child_patches=child_patches, agg=agg, **kwargs
        )
        return s_zoom - s_stop

    def inference(self, s_stop, s_zoom):
        if s_stop <= 0 and s_zoom <= 0:
            return 0
        return 1 if (s_zoom >= s_stop) else 0


class TextAlignScore(PatchScoreModule):
    def __init__(self, weight=1.0, embedder=None, k=3):
        self.weight = weight
        self.embedder = embedder
        self.k = k

    def compute_stop(self, parent_patch=None, **kwargs):
        s = self.embedder.text_sim(parent_patch, aggregate=self.agg)
        return self.weight * s

    def compute_zoom(self, parent_patch=None, child_patches=None, agg="mean", **kwargs):
        if not child_patches or len(child_patches) == 0:
            return 0.0

        s_total = 0.0
        for p in child_patches:
            s_patch = self.embedder.text_sim(p, aggregate=agg)
            s_total += s_patch

        if agg == "mean":
            s = s_total / len(child_patches)
        elif agg == "max":
            s = max(s_total / len(child_patches))
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

        return self.weight * s

    def compute_diff(self, parent_patch=None, child_patches=None, **kwargs):
        if parent_patch is None or not child_patches or len(child_patches) == 0:
            return 0.0

        s_stop = self.compute_stop(parent_patch=parent_patch, **kwargs)
        s_zoom = self.compute_zoom(
            parent_patch=parent_patch, child_patches=child_patches, **kwargs
        )
        return s_zoom - s_stop

    def inference(self, s_stop, s_zoom):
        return 1 if (s_zoom >= s_stop) else 0


class TissuePresenceScore(PatchScoreModule):
    """
    Rewards tissue patches, penalizes blank.
    """

    def __init__(self, weight=1.0, blank_thr=230):
        self.weight = weight
        self.blank_thr = blank_thr

    def _is_blank(self, patch):
        return np.array(patch).mean() > self.blank_thr

    def compute_stop(self, parent_patch=None, **kwargs):
        if not self._is_blank(parent_patch):
            return self.weight
        return 0.0

    def compute_zoom(self, parent_patch=None, child_patches=None, agg="any", **kwargs):
        if agg == "any":
            for p in child_patches:
                if not self._is_blank(p):
                    return self.weight
            return 0.0
        elif agg == "all":
            for p in child_patches:
                if self._is_blank(p):
                    return 0.0
            return self.weight
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

    def compute_diff(self, action=None, parent_patch=None, **kwargs):
        s_stop = self.compute_stop(parent_patch=parent_patch, **kwargs)
        s_zoom = self.compute_zoom(action=action, parent_patch=parent_patch, **kwargs)
        return s_zoom - s_stop

    def inference(self, s_stop, s_zoom):
        return 1 if (s_zoom > s_stop) else 0


class TissuePresencePenalty(PatchScoreModule):
    """
    Rewards tissue patches, penalizes blank.
    """

    def __init__(self, weight=1.0, blank_thr=230):
        self.weight = weight
        self.blank_thr = blank_thr

    def _is_blank(self, patch):
        return np.array(patch).mean() > self.blank_thr

    def compute_stop(self, parent_patch=None, **kwargs):
        if not self._is_blank(parent_patch):
            return self.weight
        return -self.weight

    def compute_zoom(self, parent_patch=None, child_patches=None, agg="any", **kwargs):
        if agg == "any":
            for p in child_patches:
                if not self._is_blank(p):
                    return self.weight
            return -self.weight
        elif agg == "all":
            for p in child_patches:
                if self._is_blank(p):
                    return -self.weight
            return self.weight
        else:
            raise ValueError(f"Unknown aggregation method: {agg}")

    def compute_diff(self, action=None, parent_patch=None, **kwargs):
        s_stop = self.compute_stop(parent_patch=parent_patch, **kwargs)
        s_zoom = self.compute_zoom(action=action, parent_patch=parent_patch, **kwargs)
        return s_zoom - s_stop

    def inference(self, s_stop, s_zoom):
        return 1 if (s_zoom > s_stop) else 0


# ============================================================
#  Constant / Linear Rewards
# ============================================================


class LinearReward(RewardModule):
    """
    Adds a constant offset to the reward.
    Useful for centering rewards around 0.
    Only applies on ZOOM actions.
    """

    def __init__(self, offset=0.0):
        self.offset = offset

    def compute(self, **kwargs):
        return self.offset


# ============================================================
#  A. Single-Modal Vision Rewards
# ============================================================


class InfoGainReward(RewardModule):
    r"""
    Single-modal image reward (parent vs children in PLIP image space):
    r = α (mean( cos(e_parent, e_child) ) - const)
    """

    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        # No reward if STOP action
        if action == 0:
            return 0.0

        # Require child patches to compute info-gain
        if len(child_patches) == 0:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)  # (d,)

        sims = []
        for p in child_patches:
            ec = self.embedder.img_emb(p)
            sim_t = cosine_similarity(ep, ec, dim=0)
            sim_val = (
                float(sim_t.mean().item()) if sim_t.numel() > 1 else float(sim_t.item())
            )
            sims.append(sim_val)

        # Information gain: reward dissimilarity (1 - similarity)
        return self.weight * (1.0 - np.mean(sims))


class CosineSimilarityReward(RewardModule):
    r"""
    r = e_parent^T e_mean_children
    """

    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        # No reward if STOP action
        if action == 0:
            return 0.0

        # Require child patches to compute info-gain
        if len(child_patches) == 0:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)
        ecs = [self.embedder.img_emb(p) for p in child_patches]
        e_mean = torch.stack(ecs, dim=0).mean(dim=0)

        sim_t = cosine_similarity(ep, e_mean, dim=0)
        sim_val = (
            float(sim_t.mean().item()) if sim_t.numel() > 1 else float(sim_t.item())
        )
        return self.weight * sim_val


class MaxDifferenceRangeInfoReward(RewardModule):
    r"""
    Rewards high contrast range among children relative to parent:
    r = β (max_i s_i - min_j s_j)
    where s_i = cos(e_parent, e_child_i)
    """

    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute(self, parent_patch=None, child_patches=None, **kwargs):
        if not child_patches or len(child_patches) == 0:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)

        sims = []
        for p in child_patches:
            ec = self.embedder.img_emb(p)
            sim_t = cosine_similarity(ep, ec, dim=0)
            sim_val = (
                float(sim_t.mean().item()) if sim_t.numel() > 1 else float(sim_t.item())
            )
            sims.append(sim_val)

        return self.weight * (max(sims) - min(sims))


class MaxDifferenceInfoGainReward(RewardModule):
    r"""
    Rewards maximum difference (minimum similarity) between parent and any child:
    r = α (1 - min_i cos(e_parent, e_child_i))
    """

    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        # No reward if STOP action
        if action == 0:
            return 0.0

        if len(child_patches) == 0:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)

        sims = []
        for p in child_patches:
            ec = self.embedder.img_emb(p)
            sim_t = cosine_similarity(ep, ec, dim=0)
            sim_val = (
                float(sim_t.mean().item()) if sim_t.numel() > 1 else float(sim_t.item())
            )
            sims.append(sim_val)

        min_sim = min(sims)
        return self.weight * (1.0 - min_sim)


class EntropyGainReward(RewardModule):
    r"""
    Mean entropy gain:
    r = γ (mean_i H(C_i) - H(X_k))
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def _entropy(self, img_np):
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = np.maximum(hist, 1e-12)
        return float(-(hist * np.log(hist)).sum())

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        if not child_patches or len(child_patches) == 0:
            return 0.0

        Hp = self._entropy(np.array(parent_patch))
        Hc = np.max([self._entropy(np.array(cp)) for cp in child_patches])

        if action == 0:
            return self.weight * Hp
        return self.weight * Hc


class MaxEntropyGainReward(RewardModule):
    r"""
    Maximum entropy gain (instead of mean):
    r = γ (max_i H(C_i) - H(X_k))
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def _entropy(self, img_np):
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        hist = np.maximum(hist, 1e-12)
        return float(-(hist * np.log(hist)).sum())

    def compute(self, parent_patch=None, child_patches=None, **kwargs):
        if not child_patches or len(child_patches) == 0:
            return 0.0

        Hp = self._entropy(np.array(parent_patch))
        Hc = max([self._entropy(np.array(cp)) for cp in child_patches])

        return self.weight * (Hc - Hp)


class TissuePresenceReward(RewardModule):
    """
    Rewards tissue patches, penalizes blank.
    """

    def __init__(self, weight=1.0, blank_thr=230):
        self.weight = weight
        self.blank_thr = blank_thr

    def _is_blank(self, patch):
        return np.array(patch).mean() > self.blank_thr

    def compute(self, action=None, parent_patch=None, **kwargs):
        if parent_patch is None:
            return 0.0
        if action == 0:
            return 0.0
        if self._is_blank(parent_patch) and action == 1:
            return -self.weight
        return +self.weight


class TissuePresencePenalty(RewardModule):
    """
    Rewards tissue patches, penalizes blank.
    """

    def __init__(self, weight=1.0, blank_thr=230):
        self.weight = weight
        self.blank_thr = blank_thr

    def _is_blank(self, patch):
        return np.array(patch).mean() > self.blank_thr

    def compute(self, action=None, parent_patch=None, **kwargs):
        if parent_patch is None:
            return 0.0
        if self._is_blank(parent_patch) and action == 1:
            return -self.weight
        return 0.0


# ============================================================
#  B. Multimodal Rewards (PLIP + FAISS)
# ============================================================
class TextAlignReward(RewardModule):
    """
    FAISS-based text alignment reward.
    Uses PLIP to retrieve top-k similar texts from corpus and computes similarity.
    Returns absolute parent similarity score.
    """

    def __init__(self, weight=1.0, embedder=None, k=3, aggregate="mean"):
        self.weight = weight
        self.embedder = embedder
        self.k = k
        self.aggregate = aggregate

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        if action == 0:
            s = self.embedder.text_sim(parent_patch, aggregate=self.aggregate)
        else:
            if not child_patches or len(child_patches) == 0:
                return 0.0
            s_total = 0.0
            for p in child_patches:
                s_patch = self.embedder.text_sim(p, aggregate=self.aggregate)
                s_total += s_patch
            s = s_total / len(child_patches)

        return self.weight * s


class SemanticSurpriseReward(RewardModule):
    r"""
    Multimodal "surprise" based on FAISS+PLIP scores.

    Let s(x) = faiss.get_faiss_score(PLIP(x)).

    We define:
        r = δ * | mean(softmax(child_scores)) - softmax(parent_score) |

    (Not super principled, but captures change in semantic alignment.)
    """

    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute(self, parent_patch=None, child_patches=None, **kwargs):
        if not child_patches or len(child_patches) == 0:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)
        s_parent = self.embedder.plip_sim(ep)

        child_scores = [
            self.embedder.plip_sim(self.embedder.img_emb(p)) for p in child_patches
        ]

        sp = np.array([s_parent], dtype=np.float32)
        sc = np.array(child_scores, dtype=np.float32)

        # softmax-like normalization
        ep_ = np.exp(sp - sp.max())
        ep_ /= ep_.sum() + 1e-8
        ec_ = np.exp(sc - sc.max())
        ec_ /= ec_.sum() + 1e-8

        return self.weight * float(np.abs(ec_.mean() - ep_.mean()))


# ============================================================
#  C. Structural Penalties
# ============================================================


class PatchCost(RewardModule):
    """
    Simple zoom cost: r = -λ if action == ZOOM
    """

    def __init__(self, cost=0.001):
        self.cost = cost

    def compute(self, action=None, **kwargs):
        return -self.cost if action == 1 else 0.0


class SimpleStopPenalty(RewardModule):
    """
    Simple constant penalty/reward for stopping (no FAISS dependency).

    Positive penalty = penalty for stopping = encourages zooming
    Negative penalty = reward for stopping = encourages stopping
    """

    def __init__(self, penalty=0.5):
        self.penalty = penalty

    def compute(self, action=None, curr_level=None, **kwargs):
        if action != 0 or curr_level == 0:
            return 0.0
        return -self.penalty


class DepthPenalty(RewardModule):
    def __init__(self, weight=0.01):
        self.weight = weight

    def compute(self, curr_level=None, max_level=None, **kwargs):
        return -self.weight * (max_level - curr_level)


class ZoomBudgetPenalty(RewardModule):
    def __init__(self, weight=1.0, budget=3):
        self.weight = weight
        self.budget = budget

    def compute(self, zoom_count=None, **kwargs):
        return (
            -self.weight
            if (zoom_count is not None and zoom_count > self.budget)
            else 0.0
        )


class InvalidPatchPenalty(RewardModule):
    def __init__(self, penalty=1.0):
        self.penalty = penalty

    def compute(self, invalid=False, **kwargs):
        return -self.penalty if invalid else 0.0


# ============================================================
#  Reward Engine
# ============================================================


class RewardEngine:
    def __init__(self, modules, embedder_model="plip"):
        self.modules = modules
        for m in self.modules:
            if hasattr(m, "embedder"):
                m.embedder = Embedder(img_backend=embedder_model)

    def compute(self, **kwargs):
        total = 0.0
        for m in self.modules:
            total += m.compute(**kwargs)
        return float(total)
