# A_bandit_reinforce.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, 2)  # STOP / ZOOM
        )

    def forward(self, x):
        return self.net(x)


class BanditAgent:
    def __init__(self, state_dim, lr=1e-4):
        self.policy = Policy(state_dim)
        self.opt = optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline = 0.0
        self.beta = 0.01  # baseline update speed

    def step(self, state, reward):
        logits = self.policy(state)
        dist = Categorical(logits=logits)
        action = dist.sample()

        # baseline
        self.baseline = (1 - self.beta) * self.baseline + self.beta * reward
        advantage = reward - self.baseline

        loss = -dist.log_prob(action) * advantage

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        return action.item(), loss.item()


def train_bandit_supervised(
    policy,
    optimizer,
    env,
    n_steps=100_000,
    device="cpu",
):
    policy.train()

    for step in range(n_steps):
        # 1. Sample a state
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out
        state = torch.tensor(state, dtype=torch.float32, device=device)

        # 2. Compute rewards for BOTH actions
        s_stop = env.patch_score_module.compute_stop()
        s_zoom = env.patch_score_module.compute_zoom()

        # 3. Target = argmax reward
        target = 1 if s_zoom > s_stop else 0
        target = torch.tensor([target], device=device)

        # 4. Forward
        logits = policy(state.unsqueeze(0))

        # 5. Loss
        loss = F.cross_entropy(logits, target)

        # 6. Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000 == 0:
            print(
                f"[{step}] loss={loss.item():.4f} "
                f"s_stop={s_stop:.3f} s_zoom={s_zoom:.3f}"
            )


if __name__ == "__main__":
    policy = Policy()
