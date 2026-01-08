import numpy as np
import faiss as _faisslib


class FAISS:
    def __init__(self, faiss_index_path, faiss_texts_path):
        try:
            self.index = _faisslib.read_index(faiss_index_path)
        except AttributeError:
            # Fallback: some faiss builds expose only deserialize_index
            with open(faiss_index_path, "rb") as f:
                index_bytes = f.read()
            self.index = _faisslib.deserialize_index(index_bytes)
        self.text_ids = np.load(faiss_texts_path, allow_pickle=True)

    def get_faiss_score(self, emb):
        D, I = self.index.search(emb, k=1)
        return float(D[0, 0]), self.text_ids[I[0, 0]]
