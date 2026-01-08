import os
import time
import glob
import requests
import json
import numpy as np
from tqdm import tqdm
from bs4 import BeautifulSoup

import torch
from transformers import CLIPProcessor, CLIPModel
import faiss_util


BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


class TextDownloader:
    def __init__(
        self,
        TCGA_KEYWORDS,
        out_dir="./data/epmc_tcga_corpus",
        per_project=50,
        build_faiss=False,
        faiss_index_out="txt_index.faiss",
        faiss_paths_out="filenames.npy",
    ):
        self.TCGA_KEYWORDS = TCGA_KEYWORDS
        self.out_dir = out_dir
        self.per_project = per_project

        self.build_faiss_flag = build_faiss
        self.faiss_index_out = faiss_index_out
        self.faiss_paths_out = faiss_paths_out

        os.makedirs(self.out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _query_epmc(self, keyword_query, cursor="*", page_size=25):
        params = {
            "query": f"({keyword_query}) AND OPEN_ACCESS:Y",
            "format": "json",
            "pageSize": page_size,
            "cursorMark": cursor,
        }
        r = requests.get(BASE_URL, params=params)
        r.raise_for_status()
        return r.json()

    def _fetch_fulltext_xml(self, pmcid):
        url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML"
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.text

    def _extract_snippets(self, xml_text):
        soup = BeautifulSoup(xml_text, "xml")
        snippets = []

        # Figure captions
        for fig in soup.find_all("fig"):
            cap = fig.find("caption")
            if cap:
                snippets.append(cap.get_text(" ", strip=True))

        # Histopathology / microscopy / results
        for sec in soup.find_all("sec"):
            title = sec.find("title")
            title_text = title.get_text().lower() if title else ""
            if any(k in title_text for k in ["histopath", "microscop", "result"]):
                snippets.append(sec.get_text(" ", strip=True))

        return snippets

    # ---------------------------------------------------------
    # Fetch texts for one TCGA project
    # ---------------------------------------------------------
    def fetch_for_project(self, code, keyword_query):
        save_dir = os.path.join(self.out_dir, code)
        os.makedirs(save_dir, exist_ok=True)

        cursor = "*"
        downloaded = 0

        print(f"\n=== Fetching {code}: {keyword_query} ===")

        while True:
            data = self._query_epmc(keyword_query, cursor)
            results = data.get("resultList", {}).get("result", [])

            if not results:
                break

            for hit in tqdm(results, desc=f"{code} batch"):
                pmcid = hit.get("pmcid")
                if not pmcid:
                    continue

                out_path = os.path.join(save_dir, f"{pmcid}.txt")
                if os.path.exists(out_path):
                    continue

                xml_text = self._fetch_fulltext_xml(pmcid)
                if xml_text is None:
                    continue

                snippets = self._extract_snippets(xml_text)
                if not snippets:
                    continue

                text = "\n".join(snippets)
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(text)

                downloaded += 1
                time.sleep(0.2)

                if downloaded >= self.per_project:
                    return downloaded

            cursor = data.get("nextCursorMark")
            if not cursor:
                break

        return downloaded

    # ---------------------------------------------------------
    # Fetch all TCGA texts
    # ---------------------------------------------------------
    def fetch_texts(self):
        for code, query in self.TCGA_KEYWORDS.items():
            count = self.fetch_for_project(code, query)
            print(f"{code}: downloaded {count} articles")

    # ---------------------------------------------------------
    # Text loading + chunking
    # ---------------------------------------------------------
    @staticmethod
    def _load_text(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _chunk_text(text, max_chars=300, stride=250):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start : start + max_chars])
            start += stride
        return chunks

    # ---------------------------------------------------------
    # Build FAISS (PLIP text encoder)
    # ---------------------------------------------------------
    def build_faiss(self):
        print("\nCollecting text files...")
        txt_files = glob.glob(f"{self.out_dir}/**/*.txt", recursive=True)

        docs = []
        file_refs = []

        for path in txt_files:
            text = self._load_text(path)
            for chunk in self._chunk_text(text):
                docs.append(chunk[:256])
                file_refs.append(path)

        print(f"Loaded {len(txt_files)} files → {len(docs)} chunks")

        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = CLIPModel.from_pretrained("vinid/plip").to(device)
        processor = CLIPProcessor.from_pretrained("vinid/plip")
        model.eval()

        embeddings = []
        batch_size = 8

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            inputs = processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)

            with torch.no_grad():
                feats = model.get_text_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embeddings.append(feats.cpu().numpy())

        embeddings = np.vstack(embeddings).astype("float32")

        dim = embeddings.shape[1]
        index = faiss_util.IndexFlatIP(dim)
        index.add(embeddings)

        faiss_util.write_index(index, self.faiss_index_out)
        np.save(self.faiss_paths_out, np.array(file_refs))

        print(f"Saved FAISS index → {self.faiss_index_out}")
        print(f"Saved paths      → {self.faiss_paths_out}")

    # ---------------------------------------------------------
    # Run full pipeline
    # ---------------------------------------------------------
    def run(self):
        self.fetch_texts()
        if self.build_faiss_flag:
            self.build_faiss()


if __name__ == "__main__":
    TCGA_KEYWORDS = {
        "TCGA-COAD": "colon adenocarcinoma OR colorectal adenocarcinoma",
        "TCGA-READ": "rectal adenocarcinoma OR colorectal adenocarcinoma",
    }

    downloader = TextDownloader(
        TCGA_KEYWORDS=TCGA_KEYWORDS,
        per_project=10,
        build_faiss=True,
    )
    downloader.run()
