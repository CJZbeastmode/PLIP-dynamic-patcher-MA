#!/usr/bin/env python3
"""
Main RL training script.
Wraps core logic from src/ma/ package modules.
"""

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from torch import optim

from ma.wsi import WSI
from ma.faiss_util import FAISS
from ma.dynamic_patch_env import DynamicPatchEnv
from ma.model_actor_critic import ActorCritic
from ma.image_downloader import ImageDownloader
from ma.text_downloader import TextDownloader

# NEW: imports for PLIP + embedding
from transformers import CLIPModel, CLIPProcessor
from ma.reward_module import EmbeddingComputer
from ma.rewards import infogain_only, info_gain_max_difference_only, info_gain_max_range_only, cos_sim_only, entropy_only, max_entropy_only, text_align_only, text_align_fixed_embedding_only


# ========================================================
# Load PLIP model + processor
# ========================================================
plip_model = CLIPModel.from_pretrained("vinid/plip")
plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
plip_model.eval()

# Create embedding computer (FAISS added later)
embedder = EmbeddingComputer(
    model=plip_model,
    processor=plip_processor,
    faiss=None,
    text_embed=None
)

reward_engine = infogain_only


# ========================================================
# Dataset initialization
# ========================================================
PROPORTIONS = {
    "TCGA-COAD": 0.308,
    "TCGA-READ": 0.262,
    "TCGA-ESCA": 0.583,
    "TCGA-STAD": 0.556,
    "TCGA-LUAD": 0.391,
    "TCGA-LUSC": 0.378,
    "TCGA-MESO": 0.623,
    "TCGA-CHOL": 0.864,
    "TCGA-LIHC": 0.703,
    "TCGA-PAAD": 0.648,
    "TCGA-UVM":  0.647,
    "TCGA-SKCM": 0.744
}

TCGA_KEYWORDS = {
    "TCGA-COAD": "colon adenocarcinoma OR colorectal adenocarcinoma",
    "TCGA-READ": "rectal adenocarcinoma OR colorectal adenocarcinoma",
    "TCGA-ESCA": "esophageal carcinoma",
    "TCGA-STAD": "stomach adenocarcinoma OR gastric cancer",
    "TCGA-LUAD": "lung adenocarcinoma",
    "TCGA-LUSC": "lung squamous cell carcinoma",
    "TCGA-MESO": "mesothelioma",
    "TCGA-CHOL": "cholangiocarcinoma OR bile duct cancer",
    "TCGA-LIHC": "liver hepatocellular carcinoma",
    "TCGA-PAAD": "pancreatic adenocarcinoma",
    "TCGA-UVM":  "uveal melanoma",
    "TCGA-SKCM": "skin cutaneous melanoma",
}

PROPORTIONS = {
    "TCGA-COAD": 0.5,
    "TCGA-READ": 0.5
}

TCGA_KEYWORDS = {
    "TCGA-COAD": "colon adenocarcinoma OR colorectal adenocarcinoma",
    "TCGA-READ": "rectal adenocarcinoma OR colorectal adenocarcinoma"
}

#image_downloader = ImageDownloader(proportions=PROPORTIONS, target_total=10)
#image_downloader.sample_cases()


# ========================================================
# Config
# ========================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

EXAMPLE_IMAGE = "test_patch_image.svs"
TRAIN_IMAGES = ["test_img_1.svs", "test_img_2.svs", "test_img_3.svs"]
INDEX = "txt_index.faiss"
TEXTS = "filenames.npy"

EPISODES_PER_WSI = 3 #50
EPOCHS = 300
GAMMA = 0.99
LR = 3e-4
ENTROPY_BETA = 0.05


# ========================================================
# Load FAISS + ENV SETUP
# ========================================================
fa = FAISS(INDEX, TEXTS)

# IMPORTANT: inject FAISS into embedder
embedder.faiss = fa


# Precompute state dimension
wsi = WSI(EXAMPLE_IMAGE)

# Assign embedder to reward modules
for module in reward_engine.modules:
    if hasattr(module, "embedder"):
        module.embedder = embedder

env = DynamicPatchEnv(wsi, reward_engine=reward_engine)

s0 = env.reset()
state_dim = len(s0)


# ========================================================
# Optimizer & Model
# ========================================================
model = ActorCritic(state_dim=state_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)


# ========================================================
# Functions
# ========================================================
def compute_returns(rewards, gamma=0.99):
    G = 0.0
    out = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.insert(0, G)
    return torch.tensor(out, dtype=torch.float32, device=device)


def run_episode(env, model, optimizer):
    logps = []
    values = []
    rewards = []

    state = env.reset()
    done = False

    while not done:
        s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(s)
        dist = Categorical(logits=logits)
        action = dist.sample()

        next_state, reward, done, _ = env.step(int(action.item()))

        logps.append(dist.log_prob(action))
        values.append(value.squeeze())
        rewards.append(reward)
        state = next_state if not done else None

    returns = compute_returns(rewards, GAMMA)
    values = torch.stack(values)
    logps = torch.stack(logps)
    advantages = returns - values.detach()

    entropy = dist.entropy().mean()
    policy_loss = -(logps * advantages).mean()
    value_loss = F.mse_loss(values, returns)
    loss = policy_loss + 0.5 * value_loss - ENTROPY_BETA * entropy

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"loss: {loss}")
    return sum(rewards)


# ========================================================
# Training Loop
# ========================================================
for epoch in range(1, EPOCHS + 1):

    print(f"\n--- EPOCH {epoch} ---")
    #cases = image_downloader.shuffled_case_list()

    for image in TRAIN_IMAGES:

        # 2. Create WSI and attach to env
        wsi = WSI(image)

        # Assign embedder to reward modules (important for each new WSI)
        for module in reward_engine.modules:
            if hasattr(module, "embedder"):
                module.embedder = embedder

        env = DynamicPatchEnv(wsi, reward_engine=reward_engine)

        # 3. Train on this WSI
        for _ in range(EPISODES_PER_WSI):
            reward = run_episode(env, model, optimizer)

        # 4. Cleanup
        del env
        del wsi
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print(f"Finished WSI {image}.")

# ========================================================
# Save model
# ========================================================
torch.save(model.state_dict(), "dynamic_patch_rl.pt")
print("Saved dynamic_patch_rl.pt")
