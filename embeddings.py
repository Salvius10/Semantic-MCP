import numpy as np
from sentence_transformers import SentenceTransformer


class VectorIndex:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        # First call downloads the model (~80MB) and caches it locally.
        # Every call after that is fully offline.
        self.model = SentenceTransformer(model)
        self.texts: list[str] = []
        self.matrix: np.ndarray | None = None

    def build(self, texts: list[str]):
        self.texts = texts
        # normalize_embeddings=True does the unit-length normalization
        # for us here, instead of us dividing by the norm manually.
        self.matrix = self.model.encode(texts, normalize_embeddings=True)

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.matrix @ q
        top_indices = np.argsort(-scores)[:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]