from .dynamic_patch_env import DynamicPatchEnv
from .faiss_util import FAISS
from .model_actor_critic import ActorCritic
import torch
from .wsi import WSI


EXAMPLE_IMAGE = "test_img_2.svs"
INDEX = "txt_index.faiss"
TEXTS = "filenames.npy"

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

actions = [] # Debug

def run_policy_once(model, state):
    """Given a state (patch encoding), return greedy RL action."""
    s = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    logits, _ = model(s)
    global actions
    actions.extend(logits)
    return torch.argmax(logits).item()   # 0=STOP, 1=ZOOM


def infer_zoom(env, model, lvl, x, y, max_depth=10):
    """
    Recursively apply RL zoom policy on a single starting patch.
    Returns: list of kept patches (PIL Images)
    """
    kept = []
    discarded = [] # Debug

    # Extract patch + state
    patch = env.wsi.get_patch(lvl, x, y)
    state = env.encode_state(patch)

    action = run_policy_once(model, state)

    if action == 0:  # STOP
        kept.append(patch)
        return kept, discarded

    # Otherwise: ZOOM
    discarded.append(patch)

    child_level = lvl - 1
    if child_level < 0 or max_depth <= 0:
        kept.append(patch)
        discarded.pop()
        return kept, discarded

    # Compute scale
    scale = env.wsi.get_scale(lvl)
    cx = int(x * scale)
    cy = int(y * scale)

    # Infer for each child region (4 quadrants)
    offsets = [(0, 0), (env.patch_size//2, 0),
               (0, env.patch_size//2), (env.patch_size//2, env.patch_size//2)]

    for dx, dy in offsets:
        nx = cx + int(dx * scale)
        ny = cy + int(dy * scale)
        child_kept, child_discarded = infer_zoom(env, model, child_level, nx, ny, max_depth=max_depth - 1)
        kept.extend(child_kept)
        discarded.extend(child_discarded)

    return kept, discarded


def infer_wsi(model, wsi, env):
    """
    Main inference over an entire WSI.
    """
    final_patches = []
    final_discarded = []
    lvl = wsi.max_level
    width, height = wsi.levels_info[lvl]["size"]

    for y in range(0, height, wsi.patch_size):
        for x in range(0, width, wsi.patch_size):
            kept, discarded = infer_zoom(env, model, lvl, x, y)
            final_patches.extend(kept)
            final_discarded.extend(discarded)

    print(f"Discarded: {len(final_discarded)}")

    return final_patches



# Load WSI + environment
wsi = WSI(EXAMPLE_IMAGE)
fa = FAISS(INDEX, TEXTS)
env = DynamicPatchEnv(wsi, fa)

s0 = env.reset()
state_dim = len(s0)

# Load trained model
model = ActorCritic(state_dim=state_dim).to(device)
model.load_state_dict(torch.load("dynamic_patch_rl.pt", map_location=device))
model.eval()

# Inference
patches = infer_wsi(model, wsi, env)

print("Collected patches:", len(patches))
#print(f"Actions: {actions}")

# Save them
#for i, p in enumerate(patches):
#    p.save(f"collected_patches/patch_{i}.png")
