import numpy as np
import torch
import cv2
from torch.nn.functional import cosine_similarity


# ============================================================
#  Shared Embedding Wrapper
# ============================================================

class EmbeddingComputer:
    """
    Computes PLIP image embeddings and text-related similarity.

    - model, processor: PLIP (CLIP-like) model + processor
    - faiss: optional FAISS index that implements get_faiss_score(emb)
             where emb is a PLIP image embedding (torch.Tensor)
    - text_embed: optional fallback text embedding (PLIP text encoder),
                  used only if faiss is None
    """

    def __init__(self, model, processor, faiss=None, text_embed=None):
        self.model = model
        self.processor = processor
        self.faiss = faiss
        self.text_embed = text_embed  # optional fallback

    def img_emb(self, patch):
        """
        Return L2-normalized PLIP image embedding as a 1D torch.Tensor.
        """
        inputs = self.processor(images=patch, return_tensors="pt")
        with torch.no_grad():
            feat = self.model.get_image_features(**inputs)
        feat = feat / feat.norm(dim=-1, keepdim=True)      # shape: (1, d)
        return feat.squeeze(0).cpu()                       # shape: (d,)

    def plip_sim(self, emb, k=3, aggregate="mean", img=None):
        """
        Compute PLIP multimodal score using image and text embeddings.

        Behavior:
        - Query FAISS to get top-k text IDs
        - Fetch corresponding text strings
        - Encode texts with PLIP text encoder
        - Compute similarity between image embedding and text embeddings
        - Return aggregated similarity score
        """
        # Ensure tensor shape (1, d)
        if isinstance(emb, torch.Tensor):
            emb = emb.unsqueeze(0) if emb.ndim == 1 else emb
        else:
            emb = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)

        # Require FAISS for text retrieval
        if self.faiss is None:
            return 0.0

        # FAISS search to get top-k text indices
        if img is not None:
            _, I = self.faiss.index.search(img.numpy().astype("float32"), k)
            text_indices = I[0].tolist()
        else:
            _, I = self.faiss.index.search(emb.numpy().astype("float32"), k)
            text_indices = I[0].tolist()

        # Fetch text strings from FAISS wrapper
        texts = []
        for idx in text_indices:
            if hasattr(self.faiss, 'texts') and idx < len(self.faiss.texts):
                texts.append(self.faiss.texts[idx])
            elif hasattr(self.faiss, 'text_ids') and idx < len(self.faiss.text_ids):
                texts.append(self.faiss.text_ids[idx])
        
        if not texts:
            return 0.0

        # Encode texts with PLIP
        text_inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute cosine similarity between image and each text
        img_emb_normalized = emb / emb.norm(dim=-1, keepdim=True)
        scores = []
        for i in range(text_features.shape[0]):
            sim = cosine_similarity(img_emb_normalized, text_features[i:i+1], dim=1)
            scores.append(float(sim.item()))

        # Aggregate scores
        if aggregate == "max":
            return float(max(scores))
        elif aggregate == "softmax":
            w = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=0)
            return float((w * torch.tensor(scores, dtype=torch.float32)).sum().item())
        else:  # mean
            return float(np.mean(scores))



# ============================================================
#  Base Class
# ============================================================

class RewardModule:
    def compute(self, **kwargs):
        raise NotImplementedError


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
        if action != 1 or not child_patches:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)  # (d,)

        sims = []
        for p in child_patches:
            ec = self.embedder.img_emb(p)
            sims.append(float(cosine_similarity(ep, ec, dim=0)))

        # parent-parent sim is 1.0, but subtracting a constant only shifts returns,
        # so we can just use mean(sims) directly or (mean - 1.0).
        return self.weight * (np.mean(sims) - 0.0)


class CosineSimilarityReward(RewardModule):
    r"""
    r = e_parent^T e_mean_children
    """

    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        if action != 1 or not child_patches:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)
        ecs = [self.embedder.img_emb(p) for p in child_patches]
        e_mean = torch.stack(ecs, dim=0).mean(dim=0)

        sim = cosine_similarity(ep, e_mean, dim=0).item()
        return self.weight * sim


class MaxDifferenceRangeInfoReward(RewardModule):
    r"""
    Rewards high contrast range among children relative to parent:
    r = β (max_i s_i - min_j s_j)
    where s_i = cos(e_parent, e_child_i)
    """

    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        if action != 1 or not child_patches:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)

        sims = []
        for p in child_patches:
            ec = self.embedder.img_emb(p)
            sims.append(float(cosine_similarity(ep, ec, dim=0)))

        return self.weight * (max(sims) - min(sims))
    

class MaxDifferenceInfoReward(RewardModule):
    r"""
    Rewards maximum difference (minimum similarity) between parent and any child:
    r = α (1 - min_i cos(e_parent, e_child_i))
    """

    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        if action != 1 or not child_patches:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)

        sims = []
        for p in child_patches:
            ec = self.embedder.img_emb(p)
            sims.append(float(cosine_similarity(ep, ec, dim=0)))

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
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist = hist / hist.sum()
        hist = np.maximum(hist, 1e-12)
        return float(-(hist * np.log(hist)).sum())

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        if action != 1 or not child_patches:
            return 0.0

        Hp = self._entropy(np.array(parent_patch))
        Hc = np.mean([self._entropy(np.array(cp)) for cp in child_patches])

        return self.weight * (Hc - Hp)


class MaxEntropyGainReward(RewardModule):
    r"""
    Maximum entropy gain (instead of mean):
    r = γ (max_i H(C_i) - H(X_k))
    """

    def __init__(self, weight=1.0):
        self.weight = weight

    def _entropy(self, img_np):
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray],[0],None,[256],[0,256])
        hist = hist / hist.sum()
        hist = np.maximum(hist, 1e-12)
        return float(-(hist * np.log(hist)).sum())

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        if action != 1 or not child_patches:
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

    def compute(self, parent_patch=None, action=None, **kwargs):
        if parent_patch is None:
            return 0.0
        if self._is_blank(parent_patch):
            return -self.weight
        return +self.weight


# ============================================================
#  B. Multimodal Rewards (PLIP + FAISS)
# ============================================================

class TextAlignReward(RewardModule):
    r"""
    Multimodal (PLIP + FAISS) alignment reward.

    Let s(x) be the FAISS+PLIP score for patch x:
        s(x) = faiss.get_faiss_score( PLIP(x) )

    Then:
        r = γ ( mean_i s(child_i) - s(parent) )
    """

    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        if action != 1 or parent_patch is None or not child_patches:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)
        s_parent = self.embedder.plip_sim(ep)

        child_scores = []
        for p in child_patches:
            ec = self.embedder.img_emb(p)
            child_scores.append(self.embedder.plip_sim(ec))

        return self.weight * (np.mean(child_scores) - s_parent)


class TextAlignFixedEmbeddingReward(RewardModule):
    r"""
    Multimodal (PLIP + FAISS) alignment reward.

    Let s(x) be the FAISS+PLIP score for patch x:
        s(x) = faiss.get_faiss_score( PLIP(x) )

    Then:
        r = γ ( mean_i s(child_i) - s(parent) )
    """

    def __init__(self, weight=1.0, embedder=None):
        self.weight = weight
        self.embedder = embedder

    def compute(self, action=None, parent_patch=None, child_patches=None, img=None, **kwargs):
        if action != 1 or parent_patch is None or not child_patches or img is None:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)
        s_parent = self.embedder.plip_sim(ep, img=img)

        child_scores = []
        for p in child_patches:
            ec = self.embedder.img_emb(p)
            child_scores.append(self.embedder.plip_sim(ec))

        return self.weight * (np.mean(child_scores) - s_parent)
    

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

    def compute(self, action=None, parent_patch=None, child_patches=None, **kwargs):
        if action != 1 or parent_patch is None or not child_patches:
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
        ep_ /= (ep_.sum() + 1e-8)
        ec_ = np.exp(sc - sc.max())
        ec_ /= (ec_.sum() + 1e-8)

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


class StopPenalty(RewardModule):
    """
    STOP reward using FAISS+PLIP score:

        r = s(parent) - λ_stop

    where s(parent) = faiss.get_faiss_score(PLIP(parent)).
    """

    def __init__(self, penalty=0.25, embedder=None):
        self.penalty = penalty
        self.embedder = embedder

    def compute(self, action=None, parent_patch=None, **kwargs):
        if action != 0 or parent_patch is None:
            return 0.0

        ep = self.embedder.img_emb(parent_patch)
        s_parent = self.embedder.plip_sim(ep)

        return float(s_parent - self.penalty)


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
        return -self.weight if (zoom_count is not None and zoom_count > self.budget) else 0.0


class InvalidPatchPenalty(RewardModule):
    def __init__(self, penalty=1.0):
        self.penalty = penalty

    def compute(self, invalid=False, **kwargs):
        return -self.penalty if invalid else 0.0


# ============================================================
#  Reward Engine
# ============================================================

class RewardEngine:
    def __init__(self, modules):
        self.modules = modules

    def compute(self, **kwargs):
        total = 0.0
        for m in self.modules:
            total += m.compute(**kwargs)
        return float(total)
