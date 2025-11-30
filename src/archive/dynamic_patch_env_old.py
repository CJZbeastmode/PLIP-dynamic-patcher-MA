import numpy as np
import torch
from torch.distributions import Categorical


class DynamicPatchEnv:
    """
    RL env using your WSI and FAISS classes.
    Actions:
        0 = STOP
        1 = ZOOM
    """

    def __init__(
        self,
        wsi,
        faiss,
        patch_size=256,
        max_steps=8,
        lambda_patch=0.001,  # Zoom penalty TODO tune, 
        lambda_stop=0.25,    # Stop penalty TODO tune
    ):
        self.wsi = wsi
        self.faiss = faiss
        self.patch_size = patch_size
        self.max_steps = max_steps
        self.lambda_patch = lambda_patch
        self.lambda_stop = lambda_stop

        self.max_level = wsi.max_level
        self.min_level = 0

        # State vars
        self.curr_level = None
        self.curr_x = None
        self.curr_y = None
        self.curr_emb = None
        self.curr_sim = None
        self.steps = 0
        

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _sample_root(self):
        """Pick random patch on coarsest level (scalar x,y)."""
        lvl = self.max_level
        W, H = self.wsi.levels_info[lvl]["size"]

        x = np.random.randint(0, max(1, W - self.patch_size))
        y = np.random.randint(0, max(1, H - self.patch_size))
        return lvl, x, y

    def _get_state(self):
        """Return state vector = [embedding || level_norm || x_norm || y_norm]."""
        W, H = self.wsi.levels_info[self.curr_level]["size"]

        level_norm = self.curr_level / max(1, self.max_level)
        x_norm = self.curr_x / W
        y_norm = self.curr_y / H

        extra = np.array([level_norm, x_norm, y_norm], dtype=np.float32)
        return np.concatenate([self.curr_emb, extra], axis=0).astype(np.float32)
    

    # ---------------------------------------------------------
    # Get state
    # ---------------------------------------------------------
    def encode_state(self, patch, lvl=None, x=None, y=None):
        """
        Safe version for standalone inference.
        Allows manually specifying lvl/x/y.
        """

        if lvl is not None:
            self.curr_level = lvl
        if x is not None:
            self.curr_x = x
        if y is not None:
            self.curr_y = y

        emb = self.wsi.get_emb(patch).cpu().numpy().flatten()

        W, H = self.wsi.levels_info[self.curr_level]["size"]
        level_norm = self.curr_level / max(1, self.max_level)
        x_norm = self.curr_x / W
        y_norm = self.curr_y / H

        extra = np.array([level_norm, x_norm, y_norm], dtype=np.float32)
        return np.concatenate([emb, extra], axis=0)


    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------
    def reset(self):
        self.steps = 0
        self.curr_level, self.curr_x, self.curr_y = self._sample_root()

        patch = self.wsi.get_patch(self.curr_level, self.curr_x, self.curr_y)
        emb = self.wsi.get_emb(patch)
        sim, _ = self.faiss.get_faiss_score(emb)

        self.curr_emb = emb.cpu().numpy().flatten()
        self.curr_sim = sim

        return self._get_state()

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------
    def step(self, action):
        """
        Returns:
            next_state, reward, done, info
        """
        done = False

        # STOP or lowest level → terminal
        if action == 0 or self.curr_level <= self.min_level:
            reward = float(self.curr_sim - self.lambda_stop)
            return None, reward, True, {}

        # ---------------------------------------------
        # ZOOM
        # ---------------------------------------------
        parent_level = self.curr_level
        child_level = parent_level - 1

        scale = self.wsi.get_scale(parent_level)
        cx = int(self.curr_x * scale)
        cy = int(self.curr_y * scale)

        child_sims = []
        child_embs = []
        child_coords = []

        # Four child sub-patches
        for dy in [0, self.patch_size // 2]:
            for dx in [0, self.patch_size // 2]:
                child_x = cx + int(dx * scale)
                child_y = cy + int(dy * scale)

                try:
                    patch = self.wsi.get_patch(child_level, child_x, child_y)
                except:
                    continue

                emb = self.wsi.get_emb(patch)
                sim, _ = self.faiss.get_faiss_score(emb)

                child_sims.append(sim)
                child_embs.append(emb.cpu().numpy().flatten())
                child_coords.append((child_x, child_y))

        # If no valid children → stop early
        if len(child_sims) == 0:
            reward = float(-self.lambda_patch)
            return None, reward, True, {}

        child_sims = np.array(child_sims)
        mean_sim = float(child_sims.mean())

        # reward = gain - cost
        reward = float(10 * (mean_sim - self.curr_sim) - self.lambda_patch)

        # pick best child
        idx = int(child_sims.argmax())
        self.curr_level = child_level
        self.curr_x, self.curr_y = child_coords[idx]
        self.curr_emb = child_embs[idx]
        self.curr_sim = float(child_sims[idx])

        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {}
