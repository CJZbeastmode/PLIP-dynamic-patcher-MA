import os
import time
import glob
import requests
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup
import faiss

import torch
from transformers import CLIPProcessor, CLIPModel


class TextDownloader:
    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, index, out_dir="epmc_tcga_corpus", per_project=50):
        self.TCGA_KEYWORDS = index
        self.out_dir = out_dir
        self.per_project = per_project

        os.makedirs(self.out_dir, exist_ok=True)


    # =====================================================================
    # Download text for one project
    # =====================================================================
    def fetch_for_project(self, code, keyword_query):
        """
        Download up to per_project fulltext files from EuropePMC.
        Save into out_dir/<TCGA>/
        """
        save_dir = os.path.join(self.out_dir, code)
        os.makedirs(save_dir, exist_ok=True)

        cursor = "*"
        downloaded = 0
        page_size = 25

        print(f"\n=== Fetching for {code}: {keyword_query} ===")

        while True:
            params = {
                "query": f"({keyword_query}) AND OPEN_ACCESS:Y",
                "format": "json",
                "pageSize": page_size,
                "cursorMark": cursor
            }

            r = requests.get(self.BASE_URL, params=params)
            data = r.json()

            results = data.get("resultList", {}).get("result", [])
            if not results:
                break

            for hit in tqdm(results, desc=f"{code} batch"):
                pmcid = hit.get("pmcid")
                if not pmcid:
                    continue

                # Download full-text XML
                xml_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
                xml = requests.get(xml_url)
                if xml.status_code != 200:
                    continue

                soup = BeautifulSoup(xml.text, "xml")
                snippets = []

                # FIGURE CAPTIONS
                for fig in soup.find_all("fig"):
                    cap = fig.find("caption")
                    if cap:
                        snippets.append(cap.get_text(" ", strip=True))

                # HISTOPATH / RESULTS
                for sec in soup.find_all("sec"):
                    title = sec.find("title").get_text().lower() if sec.find("title") else ""
                    if any(k in title for k in ["histopath", "microscop", "result"]):
                        snippets.append(sec.get_text(" ", strip=True))

                if not snippets:
                    continue

                text = "\n".join(snippets)
                fname = os.path.join(save_dir, f"{pmcid}.txt")
                with open(fname, "w", encoding="utf-8") as f:
                    f.write(text)

                downloaded += 1
                time.sleep(0.2)

                if downloaded >= self.per_project:
                    return downloaded

            cursor = data.get("nextCursorMark")
            if not cursor:
                break

        return downloaded


    # =====================================================================
    # Download texts for all projects
    # =====================================================================
    def download_all(self):
        for code, query in self.TCGA_KEYWORDS.items():
            count = self.fetch_for_project(code, query)
            print(f"{code}: downloaded {count} articles")


    # =====================================================================
    # Load text and chunking for FAISS
    # =====================================================================
    @staticmethod
    def load_text_file(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def chunk_text(text, max_chars=300, stride=250):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start + max_chars])
            start += stride
        return chunks


    # =====================================================================
    # Build FAISS using PLIP text encoder
    # =====================================================================
    def build_faiss(self, output_index="txt_index.faiss", output_paths="filenames.npy"):
        print("\nCollecting text files...")

        txt_files = glob.glob(f"{self.out_dir}/**/*.txt", recursive=True)
        docs = []
        file_refs = []

        for path in txt_files:
            text = self.load_text_file(path)
            for chunk in self.chunk_text(text):
                docs.append(chunk[:256])  # truncate
                file_refs.append(path)

        print(f"Loaded {len(txt_files)} files → {len(docs)} chunks.")

        # =====================================================================
        # Load PLIP
        # =====================================================================
        device = (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = CLIPModel.from_pretrained("vinid/plip").to(device)
        processor = CLIPProcessor.from_pretrained("vinid/plip")
        model.eval()

        embeddings = []
        batch_size = 8

        # =====================================================================
        # Create embeddings
        # =====================================================================
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            inputs = processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(device)

            with torch.no_grad():
                feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

        embeddings = np.vstack(embeddings).astype("float32")
        print("Embedding matrix shape:", embeddings.shape)

        # =====================================================================
        # Build FAISS index
        # =====================================================================
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        print(f"Indexed {index.ntotal} embeddings (dim={dim}).")

        # Save index + file references
        faiss.write_index(index, output_index)
        np.save(output_paths, np.array(file_refs))

        print(f"Saved FAISS index → {output_index}")
        print(f"Saved filename refs → {output_paths}")

        return index, file_refs
