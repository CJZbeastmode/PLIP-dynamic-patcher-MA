import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import optim

# ========================================================
# Repo path setup
# ========================================================
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.wsi import WSI
from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.utils.embedder import Embedder
from src.utils.patch_scores import *


# ========================================================
# ---------------- TRAINING PENALTIES --------------------
# ========================================================
# These are the ONLY knobs you need to tune.

ZOOM_COST = 0.5
# Penalty applied every time the agent chooses ZOOM

DEPTH_COST = 0.1
# Multiplied by env.zoom_count (discourages deep zoom chains)

MAX_ZOOM_FRAC = 0.5
# Soft target: fraction of steps that should be zooms

OVERZOOM_PENALTY = 1.0
# Strength of penalty if zoom frequency exceeds MAX_ZOOM_FRAC
# Set to 0.0 to disable


# ========================================================
# Actor–Critic Model
# ========================================================
class ActorCritic(nn.Module):
    """
    Actor–Critic for STOP / ZOOM.
    """

    def __init__(self, state_dim, hidden=256, n_heads=4, n_layers=1):
        super().__init__()

        self.input_proj = nn.Linear(state_dim, hidden)
        self.input_norm = nn.LayerNorm(hidden)

        enc = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=n_layers)

        self.actor = nn.Linear(hidden, 2)  # STOP / ZOOM
        self.critic = nn.Linear(hidden, 1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x):
        h = self.input_norm(self.input_proj(x))
        h = self.transformer(h.unsqueeze(1))[:, 0]
        return self.actor(h), self.critic(h)


# ========================================================
# Utility
# ========================================================
def compute_returns(rewards, gamma, device):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    return torch.tensor(list(reversed(out)), dtype=torch.float32, device=device)


# ========================================================
# Single Episode Rollout + Update
# ========================================================
def run_episode(
    env, model, optimizer, device, gamma, entropy_beta, deterministic=False
):

    logps, values, rewards, entropies = [], [], [], []

    state = env.reset()
    done = False
    step_idx = 0
    # Print table header for per-step logging
    print(
        "Step | Level | X | Y | Action | Prob | Logit_STOP | Logit_ZOOM | Value | Reward | s_stop | s_zoom | s_diff"
    )

    while not done:
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(s)

        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=1) if deterministic else dist.sample()

        # Capture previous location for logging (env may update coords on ZOOM)
        prev_level = getattr(env, "curr_level", None)
        prev_x = getattr(env, "curr_x", None)
        prev_y = getattr(env, "curr_y", None)

        next_state, score_diff, done, info = env.step(int(action.item()))

        # Compute selected action probability
        try:
            prob = float(torch.exp(dist.log_prob(action)).squeeze().cpu().item())
        except Exception:
            prob = float(torch.exp(dist.log_prob(action)).squeeze().item())

        # Extract logits and value for logging
        l0 = float(logits[0, 0].detach().cpu().item())
        l1 = float(logits[0, 1].detach().cpu().item())
        v = float(value.squeeze().detach().cpu().item())

        # Extract score module outputs if present
        s_stop = info.get("s_stop") if isinstance(info, dict) else None
        s_zoom = info.get("s_zoom") if isinstance(info, dict) else None
        s_diff = info.get("s_diff") if isinstance(info, dict) else None

        # Print one-line table entry for this step
        print(
            f"{step_idx:4d} | {prev_level!s:5s} | {prev_x!s:3s} | {prev_y!s:3s} | {info.get('action', 'NA'):6s} | {prob:0.4f} | {l0:0.3f} | {l1:0.3f} | {v:0.3f} | {score_diff:0.4f} | {s_stop!s} | {s_zoom!s} | {s_diff!s}"
        )
        step_idx += 1

        # --------------------------------------------------
        # Training-time penalties
        # --------------------------------------------------
        penalty = 0.0

        # Zoom cost
        if action.item() == 1:
            penalty -= ZOOM_COST

        # Depth cost
        penalty -= DEPTH_COST * env.zoom_count

        # Over-zoom soft penalty
        if len(rewards) > 0:
            zoom_ratio = env.zoom_count / len(rewards)
            if zoom_ratio > MAX_ZOOM_FRAC:
                penalty -= OVERZOOM_PENALTY * (zoom_ratio - MAX_ZOOM_FRAC)

        reward = score_diff + penalty

        # --------------------------------------------------

        logps.append(dist.log_prob(action).squeeze())
        values.append(value.squeeze())
        entropies.append(dist.entropy().squeeze())
        rewards.append(float(reward))

        state = next_state if not done else None

    # --------------------------------------------------
    # Compute losses
    # --------------------------------------------------
    returns = compute_returns(rewards, gamma, device)
    values_t = torch.stack(values)
    logps_t = torch.stack(logps)
    entropy = torch.stack(entropies).mean()

    advantages = returns - values_t.detach()
    if advantages.numel() > 1:
        adv_std = advantages.std(unbiased=False)
        if adv_std > 1e-6:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)

    policy_loss = -(logps_t * advantages).mean()
    value_loss = F.mse_loss(values_t, returns)
    loss = policy_loss + 0.5 * value_loss - entropy_beta * entropy

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    print(
        f"Episode finished | "
        f"Return: {sum(rewards):.4f} | "
        f"Steps: {len(rewards)} | "
        f"Loss: {loss.item():.4f} | "
        f"Probability of ZOOM: {env.zoom_count / max(1, len(rewards)):.4f}"
    )

    return float(sum(rewards)), len(rewards), float(loss.item())


# ========================================================
# Main
# ========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trained-images-prefix", type=str, default="data/images")
    parser.add_argument("--score-module", type=str, default="img_sim_score")
    parser.add_argument("--img-train-count", type=int, default=60)
    parser.add_argument("--episodes-per-wsi", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--output", type=str, default="data/models")
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    gamma = 0.99
    lr = 1e-4
    entropy_beta = 0.02

    # score_module = PATCH_SCORE_MODULES[args.score_module]

    embedder = Embedder(img_backend="plip")

    all_images = sorted(
        f for f in os.listdir(args.trained_images_prefix) if f.endswith(".svs")
    )
    train_images = all_images[: args.img_train_count]

    if args.test_run:
        train_images = train_images[:1]
        args.episodes_per_wsi = 1
        args.epochs = 1

    wsi0 = WSI(os.path.join(args.trained_images_prefix, train_images[0]))
    env0 = DynamicPatchEnv(wsi0, patch_score=args.score_module)
    state_dim = len(env0.reset())

    model = ActorCritic(state_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        returns_epoch = []

        for img in train_images:
            wsi = WSI(os.path.join(args.trained_images_prefix, img))
            env = DynamicPatchEnv(wsi, patch_score=args.score_module)

            for ep in range(args.episodes_per_wsi):
                ret, steps, loss = run_episode(
                    env,
                    model,
                    optimizer,
                    device,
                    gamma,
                    entropy_beta,
                    args.deterministic,
                )
                returns_epoch.append(ret)
                print(f"{img} | ep {ep+1} | return={ret:.2f} | steps={steps}")

        print(f"Epoch avg return: {np.mean(returns_epoch):.3f}")

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(args.output, "rl_a2c.pt"))


if __name__ == "__main__":
    main()
