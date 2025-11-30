"""
Medical Image Analysis (MA) package.

Core modules:
  - reward_module: PLIP-based reward functions
  - rewards: Aggregated reward engines
  - dynamic_patch_env: RL environment for patch-based navigation
  - wsi: Whole slide image utilities
  - faiss_util: FAISS index utilities
  - model_actor_critic: Actor-Critic RL model
  - inference: Inference and evaluation utilities
"""

__version__ = "0.1.0"
__all__ = [
    "reward_module",
    "rewards",
    "dynamic_patch_env",
    "wsi",
    "faiss_util",
    "model_actor_critic",
    "inference",
]
