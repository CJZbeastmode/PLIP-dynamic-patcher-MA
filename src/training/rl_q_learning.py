# B_qlearning_full.py

import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import sys
from pathlib import Path

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


# =========================
# Q NETWORK
# =========================


class QNet(nn.Module):
    """
    Q(s) -> [Q(stop), Q(zoom)]
    """

    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# AGENT
# =========================


class DQNAgent:
    def __init__(
        self,
        state_dim,
        lr=1e-4,
        gamma=0.99,
        eps=0.2,
        eps_min=0.01,
        eps_decay=0.999,
        buffer_size=50_000,
        target_update=1_000,
        device="cpu",
    ):
        self.device = device
        self.gamma = gamma

        self.q = QNet(state_dim).to(device)
        self.q_target = QNet(state_dim).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)

        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.step_count = 0

    def act(self, state):
        if random.random() < self.eps:
            return random.randint(0, 1)

        with torch.no_grad():
            q_vals = self.q(state)
            return torch.argmax(q_vals).item()

    def store(self, s, a, r, s_next, done):
        self.buffer.append(
            (
                s.detach().cpu(),
                int(a),
                float(r),
                None if s_next is None else s_next.detach().cpu(),
                float(done),
            )
        )

    """
    def train_epoch(train_images, images_dir, agent, args, device):
        epoch_returns = []

        for img in train_images:
            wsi = WSI(os.path.join(images_dir, img))
            env = DynamicPatchEnv(wsi, patch_score=args.score_module)

            for ep in range(args.episodes_per_wsi):
                ret = run_episode(
                    env=env,
                    agent=agent,
                    max_steps=args.max_steps,
                    device=device,
                )
                epoch_returns.append(ret)

                print(
                    f"{img} | ep {ep+1:02d} | "
                    f"return={ret:.3f} | eps={agent.eps:.3f}"
                )

        return epoch_returns
        """

    def train_step(self, batch_size=64):
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.stack(states).to(self.device)  # (B, D)
        actions = torch.tensor(actions, device=self.device).long()  # (B,)
        rewards = torch.tensor(rewards, device=self.device).float()  # (B,)
        dones = torch.tensor(dones, device=self.device).float()  # (B,)

        # Q(s,a)
        q_vals = self.q(states)  # (B, 2)
        q_sa = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Build q_next with masking for terminal transitions
        with torch.no_grad():
            q_next = torch.zeros(
                batch_size, device=self.device
            )  # default 0 for terminal

            non_terminal_idx = [i for i, ns in enumerate(next_states) if ns is not None]
            if len(non_terminal_idx) > 0:
                ns_batch = torch.stack([next_states[i] for i in non_terminal_idx]).to(
                    self.device
                )  # (B_nt, D)
                q_next_vals = self.q_target(ns_batch).max(1)[0]  # (B_nt,)
                q_next[torch.tensor(non_terminal_idx, device=self.device)] = q_next_vals

            target = rewards + self.gamma * (1 - dones) * q_next  # (B,)

        loss = nn.functional.mse_loss(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return loss.item()

    def decay_eps(self):
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    # =========================
    # TRAINING LOOP
    # =========================

    def train_dqn(
        self,
        env,
        n_episodes=10_000,
        max_steps=10,
        batch_size=64,
        device="cpu",
        log_info=None,
    ):
        total_rewards = []

        for ep in range(n_episodes):
            if log_info is not None:
                print(
                    f"Ep {ep + 1}/{n_episodes} | Img {log_info['image_index']}/{log_info['total_images']} | Epoch {log_info['epoch']}/{log_info['epochs']}"
                )
            else:
                print(f"Ep {ep + 1}/{n_episodes}")
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device)

            ep_reward = 0.0

            for _ in range(max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)

                if not done:
                    next_state_t = torch.tensor(
                        next_state, dtype=torch.float32, device=device
                    )
                else:
                    next_state_t = None

                self.store(state, action, reward, next_state_t, done)
                loss = self.train_step(batch_size)

                if done:
                    break

                state = next_state_t
                ep_reward += reward

            self.decay_eps()
            total_rewards.append(ep_reward)

            if ep % 100 == 0:
                print(
                    f"[EP {ep:05d}] | R={ep_reward:.3f} | eps={self.eps:.3f} | loss={loss}"
                )

        return total_rewards

    # =========================
    # INFERENCE
    # =========================

    @torch.no_grad()
    def infer(agent, state):
        state = torch.tensor(state, dtype=torch.float32, device=agent.device)
        q_vals = agent.q(state)
        action = torch.argmax(q_vals).item()
        return action, q_vals.cpu().numpy()


# =========================
# MAIN
# =========================


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", type=str, default="data/images")
    parser.add_argument("--score-module", type=str, default="text_align_score")
    parser.add_argument("--img-train-count", type=int, default=50)
    parser.add_argument("--episodes-per-wsi", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default="data/models")
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    # --------------------------------------------------
    # Collect WSIs
    # --------------------------------------------------
    all_images = sorted(f for f in os.listdir(args.images_dir) if f.endswith(".svs"))
    train_images = all_images[: args.img_train_count]

    if args.test_run:
        train_images = train_images[:1]
        args.episodes_per_wsi = 1
        args.epochs = 1

    # --------------------------------------------------
    # Initialize env once to get state_dim
    # --------------------------------------------------
    wsi0 = WSI(os.path.join(args.images_dir, train_images[0]))
    env0 = DynamicPatchEnv(wsi0, patch_score=args.score_module)
    state_dim = len(env0.reset())

    print(f"State dim: {state_dim}")

    # --------------------------------------------------
    # Create agent
    # --------------------------------------------------
    agent = DQNAgent(
        state_dim=state_dim,
        device=device,
    )

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(args.epochs):
        # print(f"\n=== Epoch {epoch + 1}/{args.epochs} ===")
        epoch_rewards = []

        img_order = 0
        for img in train_images:
            # print(f"Training on image: {img} ({img_order + 1}/{len(train_images)})")
            wsi = WSI(os.path.join(args.images_dir, img))
            env = DynamicPatchEnv(wsi, patch_score=args.score_module)

            log_info = {
                "epoch": epoch + 1,
                "epochs": args.epochs,
                "image_index": img_order + 1,
                "total_images": len(train_images),
            }
            total_rewards = agent.train_dqn(
                env,
                n_episodes=args.episodes_per_wsi,
                max_steps=args.max_steps,
                device=device,
                log_info=log_info,
            )
            epoch_rewards.extend(total_rewards)
            img_order += 1

        print(
            f"Epoch {epoch+1} avg reward: "
            f"{sum(epoch_rewards) / max(1, len(epoch_rewards)):.4f}"
        )

    # --------------------------------------------------
    # Save model
    # --------------------------------------------------
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        path = os.path.join(args.output, "dqn_zoom.pt")
        torch.save(agent.q.state_dict(), path)
        print(f"Model saved to {path}")


if __name__ == "__main__":
    main()
