import numpy as np
import os
import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from torch import optim

from wsi import WSI
from faiss_util.faiss_util import FAISS
from dynamic_patch_env import DynamicPatchEnv
from model_actor_critic import ActorCritic
from downloader.image_downloader import ImageDownloader
from downloader.text_downloader import TextDownloader


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

TCGA_KEYWORDS = {
    "TCGA-COAD": "colon adenocarcinoma OR colorectal adenocarcinoma",
    "TCGA-READ": "rectal adenocarcinoma OR colorectal adenocarcinoma"
}


PROPORTIONS = {
    "TCGA-COAD": 0.5,
    "TCGA-READ": 0.5
}


image_downloader = ImageDownloader(proportions=PROPORTIONS, target_total=10)
image_downloader.sample_cases()
#text_downloader = TextDownloader(index=TCGA_KEYWORDS)
#text_downloader.download_all()
#text_downloader.build_faiss()



# ========================================================
# Config
# ========================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

EXAMPLE_IMAGE = "test_patch_image.svs"
INDEX = "txt_index.faiss"
TEXTS = "filenames.npy"

EPISODES_PER_WSI = 50
EPOCHS = 3
GAMMA = 0.99 # Controls long term/short term rewards TODO tune
LR = 3e-4
ENTROPY_BETA = 0.05 # TODO tune


# ========================================================
# Load objects
# ========================================================
wsi = WSI(EXAMPLE_IMAGE) # TODO: address
fa = FAISS(INDEX, TEXTS)
env = DynamicPatchEnv(wsi, fa)

# Precompute state dimension
s0 = env.reset()
state_dim = len(s0)

# Optimizer
model = ActorCritic(state_dim=state_dim).to(device) # NN
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
    print("\n\n")

    return sum(rewards)



# ========================================================
# Training Loop
# ========================================================

for epoch in range(1, EPOCHS + 1):

    print(f"\n--- EPOCH {epoch} ---")

    cases = image_downloader.shuffled_case_list()

    for cid, file_id, file_name, project in cases:

        # -------- 1) Download exactly ONE WSI ------------
        wsi_path = image_downloader.download_one(cid, file_id)

        # -------- 2) Create environment ------------------
        wsi = WSI(wsi_path)
        env = DynamicPatchEnv(wsi, fa)

        # -------- 3) Train on this single WSI -------------
        for _ in range(EPISODES_PER_WSI):
            reward = run_episode(env, model, optimizer)

        # -------- 4) Cleanup ------------------------------
        del env
        del wsi
        os.remove(wsi_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Finished WSI {cid}.svs and deleted.")


# ========================================================
# Save model
# ========================================================
torch.save(model.state_dict(), "dynamic_patch_rl.pt")
print("Saved dynamic_patch_rl.pt")
