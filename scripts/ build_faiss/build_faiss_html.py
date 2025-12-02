# pip install faiss-cpu beautifulsoup4 transformers torch numpy
import faiss_util.faiss_util as faiss_util
import numpy as np
from bs4 import BeautifulSoup
from transformers import CLIPProcessor, CLIPModel
import glob
import torch

# 1) load and parse html files
def extract_text_from_html(path):
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.extract()
    return soup.get_text(separator=" ", strip=True)

def chunk_text(text, max_chars=300, stride=250):
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + max_chars])
        start += stride
    return chunks

# combined pipeline
html_files = glob.glob("text/*.html")
docs, file_refs = [], []

for path in html_files:
    text = extract_text_from_html(path)
    for chunk in chunk_text(text):
        docs.append(chunk)
        file_refs.append(path)

print(f"Loaded {len(html_files)} HTML files → created {len(docs)} text chunks.")


# 2) load PLIP (same as for image)
device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
model = CLIPModel.from_pretrained("vinid/plip").to(device)
processor = CLIPProcessor.from_pretrained("vinid/plip")
model.eval()

# 3) create text embeddings
embeddings = []
batch_size = 8

# ---- HARD TRUNCATE ----
MAX_CHARS = 256  # keeps text short; CLIP limit is ~77 tokens
docs = [t[:MAX_CHARS] for t in docs]

for i in range(0, len(docs), batch_size):
    batch = docs[i : i + batch_size]
    inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
    feats = feats / feats.norm(dim=-1, keepdim=True)
    embeddings.append(feats.cpu().numpy())

embeddings = np.vstack(embeddings).astype("float32")
print("Embedding matrix shape:", embeddings.shape)

# 4) build FAISS index
d = embeddings.shape[1]
index = faiss_util.IndexFlatIP(d)  # cosine similarity with normalized vectors
index.add(embeddings)
print(f"Indexed {index.ntotal} documents with dim {d}.")

# 5) optional: test query
query = "tumor cell morphology"
inputs = processor(text=[query], return_tensors="pt", padding=True, truncation=True).to(device)
with torch.no_grad():
    q_emb = model.get_text_features(**inputs)
q_emb = q_emb / q_emb.norm()
q_emb = q_emb.cpu().numpy().astype("float32")

D, I = index.search(q_emb, k=min(3, index.ntotal))
for rank, idx in enumerate(I[0]):
    print(f"{rank+1}. File: {file_refs[idx]}  (score={D[0][rank]:.4f})")

# 6) save
faiss_util.write_index(index, "html_index.faiss")
np.save("filenames.npy", np.array(file_refs))
