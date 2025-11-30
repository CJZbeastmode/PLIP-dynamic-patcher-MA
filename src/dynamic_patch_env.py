import numpy as np
from rewards.reward_module import InfoGainReward, PatchCost, StopPenalty, RewardEngine
import torch
from torch.distributions import Categorical


class DynamicPatchEnv:
    """
    RL env using WSI ONLY.
    Actions:
        0 = STOP
        1 = ZOOM

    IMPORTANT:
        - Env does NOT compute embeddings
        - Env does NOT compute similarities (no FAISS)
        - RewardModules handle all PLIP embedding + similarity
        - Env extracts patches ONLY
    """

    def __init__(
        self,
        wsi,
        reward_engine=None,
        patch_size=256,
        max_steps=8,
    ):
        self.wsi = wsi
        self.patch_size = patch_size
        self.max_steps = max_steps

        self.max_level = wsi.max_level
        self.min_level = 0

        # RL state
        self.curr_level = None
        self.curr_x = None
        self.curr_y = None
        self.steps = 0
        self.zoom_count = 0

        # Default reward engine (PLIP-style)
        if reward_engine is None:
            self.reward_engine = RewardEngine([
                InfoGainReward(weight=10.0),
                PatchCost(cost=0.001),
                StopPenalty(penalty=0.25),
            ])
        else:
            self.reward_engine = reward_engine

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _sample_root(self):
        lvl = self.max_level
        W, H = self.wsi.levels_info[lvl]["size"]

        x = np.random.randint(0, max(1, W - self.patch_size))
        y = np.random.randint(0, max(1, H - self.patch_size))
        return lvl, x, y

    def _get_state(self):
        """
        State is ONLY coordinates + level.
        NO embeddings here.
        """
        W, H = self.wsi.levels_info[self.curr_level]["size"]

        level_norm = self.curr_level / max(1, self.max_level)
        x_norm = self.curr_x / W
        y_norm = self.curr_y / H

        return np.array([level_norm, x_norm, y_norm], dtype=np.float32)

    # ---------------------------------------------------------
    # Safe encoding (standalone inference)
    # DOES NOT compute embeddings anymore
    # ---------------------------------------------------------
    def encode_state(self, patch, lvl=None, x=None, y=None):
        if lvl is not None:
            self.curr_level = lvl
        if x is not None:
            self.curr_x = x
        if y is not None:
            self.curr_y = y

        W, H = self.wsi.levels_info[self.curr_level]["size"]
        level_norm = self.curr_level / self.max_level
        x_norm = self.curr_x / W
        y_norm = self.curr_y / H

        extra = np.array([level_norm, x_norm, y_norm], dtype=np.float32)
        return extra   # no embeddings

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------
    def reset(self):
        self.steps = 0
        self.zoom_count = 0

        self.curr_level, self.curr_x, self.curr_y = self._sample_root()

        return self._get_state()

    # ---------------------------------------------------------
    # Step
    # ---------------------------------------------------------
    def step(self, action):
        done = False

        # Parent patch
        parent_patch = self.wsi.get_patch(self.curr_level, self.curr_x, self.curr_y)

        # ---------------------------------------------------
        # STOP or reached level 0
        # ---------------------------------------------------
        if action == 0 or self.curr_level <= self.min_level:
            reward = self.reward_engine.compute(
                action=0,
                parent_patch=parent_patch,
                child_patches=[],
                curr_level=self.curr_level,
                max_level=self.max_level,
                zoom_count=self.zoom_count,
                invalid=False,
            )
            return None, reward, True, {}

        # ---------------------------------------------------
        # ZOOM CASE
        # ---------------------------------------------------
        parent_level = self.curr_level
        child_level = parent_level - 1

        scale = self.wsi.get_scale(parent_level)
        cx = int(self.curr_x * scale)
        cy = int(self.curr_y * scale)

        child_patches = []
        child_coords = []
        invalid = False

        # 2×2 children
        for dy in [0, self.patch_size // 2]:
            for dx in [0, self.patch_size // 2]:
                child_x = cx + int(dx * scale)
                child_y = cy + int(dy * scale)

                try:
                    patch = self.wsi.get_patch(child_level, child_x, child_y)
                except:
                    invalid = True
                    continue

                child_patches.append(patch)
                child_coords.append((child_x, child_y))

        # No valid children → forced stop
        if len(child_patches) == 0:
            reward = self.reward_engine.compute(
                action=1,
                parent_patch=parent_patch,
                child_patches=[],
                curr_level=parent_level,
                max_level=self.max_level,
                zoom_count=self.zoom_count,
                invalid=True,
            )
            return None, reward, True, {}

        # Compute reward with patches only
        reward = self.reward_engine.compute(
            action=1,
            parent_patch=parent_patch,
            child_patches=child_patches,
            curr_level=parent_level,
            max_level=self.max_level,
            zoom_count=self.zoom_count,
            invalid=invalid,
        )

        # ---------------------------------------------------
        # SELECT NEXT CHILD
        # IMPORTANT CHANGE:
        #   Env no longer chooses "best" child using similarity
        #   RL must learn where to zoom → env picks child 0.
        #
        #   If you want RL to choose child among 4 options:
        #       I can extend action space for you.
        # ---------------------------------------------------
        idx = 0
        self.curr_level = child_level
        self.curr_x, self.curr_y = child_coords[idx]

        self.zoom_count += 1
        self.steps += 1

        if self.steps >= self.max_steps:
            done = True

        return self._get_state(), reward, done, {}
