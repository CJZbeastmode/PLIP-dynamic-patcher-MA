import numpy as np
import os
import sys
"""Simplified training script.

Adds src to sys.path for local imports.
Skips FAISS index if unavailable; uses purely vision reward engine.
"""
SRC_DIR = os.path.dirname(__file__)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from torch import optim

from wsi import WSI
FAISS = None  # Removed dynamic FAISS loading
from dynamic_patch_env import DynamicPatchEnv
from model_actor_critic import ActorCritic
from downloader.image_downloader import ImageDownloader
from downloader.text_downloader import TextDownloader

# NEW: imports for PLIP + embedding
from transformers import CLIPModel, CLIPProcessor
from rewards.reward_module import EmbeddingComputer
from rewards.rewards import ENGINES


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

EXAMPLE_IMAGE = "data/image/test_img_1.svs"
TRAIN_IMAGES = [
    "data/image/test_img_1.svs",
    "data/image/test_img_2.svs",
    "data/image/test_img_3.svs",
]
INDEX = "data/faiss/txt_index.faiss"  # retained for reference
TEXTS = "data/faiss/filenames.npy"

EPISODES_PER_WSI = 3 #50
EPOCHS = 300
GAMMA = 0.99
LR = 3e-4
ENTROPY_BETA = 0.05


# ========================================================
# Load FAISS + ENV SETUP
# ========================================================
# Skip FAISS loading (environment lacks required methods)
embedder.faiss = None


# Precompute state dimension
wsi = WSI(EXAMPLE_IMAGE)

# Select a default engine from catalog
reward_engine = ENGINES.get("infogain_only")

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
torch.save(model.state_dict(), "data/model/dynamic_patch_rl.pt")
print("Saved data/model/dynamic_patch_rl.pt")
