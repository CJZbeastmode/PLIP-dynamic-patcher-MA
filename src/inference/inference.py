import sys
from pathlib import Path

# Ensure repo root is on sys.path so `src` package imports work when running
# this script directly
repo_root = str(Path(__file__).resolve().parents[2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.utils.dynamic_patch_env import DynamicPatchEnv
from src.faiss_util.faiss_util import FAISS
from src.models.model_actor_critic import ActorCritic
from src.utils.wsi import WSI

import torch
import os
import webbrowser
from transformers import CLIPModel, CLIPProcessor


EXAMPLE_IMAGE = "./data/to_test_image/test_img_1.svs"
INDEX = "./data/faiss/txt_index.faiss"
TEXTS = "./data/faiss/filenames.npy"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# --------------------------------------------------------------------------
# 1. RUN MODEL
# --------------------------------------------------------------------------
def run_policy_once(model, state, stop_bias=0.0):
    """
    Return action: 0=STOP, 1=ZOOM.

    stop_bias: Added to STOP logit to make stopping more likely.
               Use positive values (e.g., 1.0, 2.0) to reduce aggressive zooming.
    """
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = model(s)

    # Apply stop bias to make stopping more attractive
    logits[0, 0] += stop_bias  # logits[0] is STOP action

    return torch.argmax(logits).item()


# --------------------------------------------------------------------------
# 2. CORRECT COORDINATE SCALING
# --------------------------------------------------------------------------
def get_child_coordinates(wsi, parent_level, parent_x, parent_y):
    """
    Convert coordinates from parent_level → child_level = parent_level - 1.

    Uses dynamic scale factor from actual level dimensions.
    Scale can be 2, 4, or other values depending on the pyramid.
    """
    child_level = parent_level - 1
    if child_level < 0:
        return None

    # Use get_scale which returns child_width / parent_width
    scale = wsi.get_scale(parent_level)

    cx = int(parent_x * scale)
    cy = int(parent_y * scale)

    return child_level, cx, cy, scale


# --------------------------------------------------------------------------
# 3. DYNAMIC PATCH SUBDIVISION
# --------------------------------------------------------------------------
def get_child_offsets(wsi, parent_level):
    """
    Get offsets for child patches using dynamic grid from WSI.

    For scale=2: returns 4 children (2x2 grid)
    For scale=4: returns 16 children (4x4 grid)
    """
    return wsi.get_child_grid(parent_level)


# --------------------------------------------------------------------------
# 4. REWRITTEN RECURSIVE ZOOM
# --------------------------------------------------------------------------
# Inference-time stop bias to control zoom aggressiveness
# Higher values = more stopping, less zooming
# Set to 0.0 to use model's raw predictions without bias
STOP_BIAS = 0.0


def infer_zoom(env, model, level, x, y, max_depth=10):
    """
    Recursive zoom with correct coordinate propagation.
    Returns:
        kept:      [(patch, meta)]
        discarded: [(patch, meta)]
    """

    kept = []
    discarded = []

    # Extract patch & encode state with coordinates
    patch = env.wsi.get_patch(level, x, y)
    state = env.encode_state(patch, lvl=level, x=x, y=y)

    action = run_policy_once(model, state, stop_bias=STOP_BIAS)
    metadata = {"level": level, "x": x, "y": y}

    # If STOP → keep this patch
    if action == 0:
        print("Stopping at level {}, x {}, y {}".format(level, x, y))
        kept.append((patch, metadata))
        return kept, discarded

    print("Zooming at level {}, x {}, y {}".format(level, x, y))

    # If ZOOM → this patch is discarded, children explored
    discarded.append((patch, metadata))

    # No deeper level available
    if level == 0 or max_depth <= 0:
        kept.append((patch, metadata))
        discarded.pop()  # not discarded since we couldn't zoom
        return kept, discarded

    # Compute child coordinates
    out = get_child_coordinates(env.wsi, level, x, y)
    child_level, cx, cy, scale = out

    # Get dynamic child offsets based on actual scale factor
    child_offsets = get_child_offsets(env.wsi, level)

    # Infer recursively
    for dx, dy in child_offsets:
        nx = cx + dx
        ny = cy + dy
        k, d = infer_zoom(env, model, child_level, nx, ny, max_depth - 1)
        kept.extend(k)
        discarded.extend(d)

    return kept, discarded


# --------------------------------------------------------------------------
# 5. FULL-SLIDE INFERENCE
# --------------------------------------------------------------------------
def infer_wsi(model, wsi, env):
    """
    Slide-wide inference at thumbnail level = wsi.max_level.
    """

    kept_all = []
    disc_all = []

    lvl = wsi.max_level
    width, height = wsi.levels_info[lvl]["size"]

    for y in range(0, height, wsi.patch_size):
        for x in range(0, width, wsi.patch_size):
            kept, disc = infer_zoom(env, model, lvl, x, y)
            kept_all.extend(kept)
            disc_all.extend(disc)

    print(f"Kept patches: {len(kept_all)}")
    print(f"Discarded:    {len(disc_all)}")

    return kept_all, disc_all


# --------------------------------------------------------------------------
# 6. MAIN SCRIPT
# --------------------------------------------------------------------------
from transformers import CLIPModel, CLIPProcessor
from src.utils.embedder import Embedder

# Load PLIP model for embeddings

wsi = WSI(EXAMPLE_IMAGE)
fa = FAISS(INDEX, TEXTS)

# Create embedder for state encoding
# embedder = EmbeddingComputer(
#    model=model,
#    processor=preprocess,
#    faiss=fa,
#    text_embed=None
# )

plip_model = CLIPModel.from_pretrained("vinid/plip")
plip_processor = CLIPProcessor.from_pretrained("vinid/plip")
plip_model.eval()

# Create embedding instance (FAISS added later)
embedder = Embedder(img_backend="plip")


env = DynamicPatchEnv(wsi, embedder=embedder)

s0 = env.reset()
state_dim = len(s0)
print(f"State dimension: {state_dim} (3 coords + 512 PLIP embedding)")

# Load RL model
model = ActorCritic(state_dim=state_dim).to(device)

# NOTE: Old models trained with state_dim=3 are incompatible!
# You must retrain with the new 515-dim state (3 coords + 512 PLIP embedding)
# Uncomment and update path once you have a retrained model:
model.load_state_dict(
    torch.load(
        "./checkpoint_epoch20_img3997.pt",
        # "./data/model/rewardEngine=infogain_only/img-count=60/episodes-per-wsi=5/epoch=5/model.pt",
        map_location=device,
    )
)
model.eval()

# Run inference
kept_patches, discarded_patches = infer_wsi(model, wsi, env)

# Save patches
os.makedirs("./data/collected_patches", exist_ok=True)

for i, (patch, meta) in enumerate(kept_patches):
    lvl = meta["level"]
    x = meta["x"]
    y = meta["y"]
    # patch.save(f"./data/collected_patches/patch_lvl{lvl}_x{x}_y{y}.png")

print(f"Saved {len(kept_patches)} kept patches.")

# Visualization
from src.utils.visualize_patches import generate_visualization

html_path = generate_visualization(wsi, kept_patches, discarded_patches)

webbrowser.open(f"file://{os.path.abspath(html_path)}")
print("Visualization opened.")
