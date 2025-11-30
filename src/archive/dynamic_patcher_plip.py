from PIL import Image
import torch
import numpy as np
import os
from transformers import CLIPProcessor, CLIPModel
from wsi import WSI
from faiss.faiss_util import FAISS


# Directories
out_dir = "patches_gain_plip_3"
os.makedirs(out_dir, exist_ok=True)


# Load PLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("vinid/plip").to(device)
processor = CLIPProcessor.from_pretrained("vinid/plip")
model.eval()

def plip_img_text_score(img_emb, text_emb):
    """
    Both embeddings must be L2-normalized.
    Score = cosine similarity (higher = more similar).
    """
    img_emb = img_emb / img_emb.norm()
    text_emb = text_emb / text_emb.norm()
    return float((img_emb @ text_emb.T).item())


def embed_text(text):
    inputs = processor(text=[text], return_tensors="pt").to(device)
    with torch.no_grad():
        t_emb = model.get_text_features(**inputs)
    return t_emb / t_emb.norm()


# Load FAISS index
faiss_index_path = "../txt_index.faiss"
faiss_texts_path = "../filenames.npy"
faiss = FAISS(faiss_index_path, faiss_texts_path)


# Load WSI object (handles synthetic levels automatically)
image_name = "test_patch_image.svs"
wsi = WSI(image_path=image_name)


# Start calibration
print("Calibration Starts")
patch_size = 256
gains = []

# Loop from coarsest → finest pyramid
for lvl in range(wsi.max_level, wsi.max_level - 1, -1): # TOMODIFY
    print(f"Current level: {lvl}")
    dims_x, dims_y = wsi.levels_info[lvl]["size"]

    # iterate patches on this level
    for y in range(0, dims_y, patch_size):
        for x in range(0, dims_x, patch_size):

            # parent patch
            parent_patch = wsi.get_patch(lvl, x, y)
            parent_img_emb = wsi.get_emb(parent_patch)
            _, parent_text = faiss.get_faiss_score(parent_img_emb)
            parent_text_emb = embed_text(parent_text)
            parent_score = plip_img_text_score(parent_img_emb, parent_text_emb)

            # if top level reached: no child
            child_level = lvl - 1
            if child_level < 0:
                continue

            # compute scale
            scale = wsi.get_scale(lvl)
            cx = int(x * scale)
            cy = int(y * scale)

            # sample 4 child patches
            child_scores = []
            for dy in [0, patch_size // 2]:
                for dx in [0, patch_size // 2]:

                    region_patch = wsi.get_patch(
                        child_level,
                        cx + int(dx * scale),
                        cy + int(dy * scale)
                    )

                    child_img_emb = wsi.get_emb(region_patch)
                    _, child_text = faiss.get_faiss_score(child_img_emb)
                    child_text_emb = embed_text(child_text)
                    child_score = plip_img_text_score(child_img_emb, child_text_emb)
                    child_scores.append(child_score)

            if not child_scores:
                continue

            mean_child_score = np.mean(child_scores)

            # Compute gain
            gain = mean_child_score - parent_score
            gains.append([x, y, gain, parent_score, mean_child_score, parent_text])

            # Save patch
            filename = f"{gain:.3f}_lvl{lvl}_x{x}_y{y}.png"
            parent_patch.save(os.path.join(out_dir, filename))

            print(f"For x={x}/{dims_x}, y={y}/{dims_y}, gain={gain}")

print(gains)
