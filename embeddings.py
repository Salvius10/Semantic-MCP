import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from ranking import reciprocal_rank_fusion

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    # Insert a boundary at lower->upper transitions so camelCase splits too
    # (readFile -> "read File"); snake_case/kebab-case already split since
    # `_`/`-` aren't in the token regex.
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class VectorIndex:
    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        # First call downloads the model (~80MB) and caches it locally.
        # After that, local_files_only skips the Hub round-trips
        # sentence-transformers otherwise makes on every load to check for
        # updates - those alone can blow past an MCP client's startup
        # timeout even though the model itself is already cached.
        try:
            self.model = SentenceTransformer(model, local_files_only=True)
        except OSError:
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


class HybridIndex:
    """Fuses vector similarity with BM25 lexical search via reciprocal
    rank fusion. The bi-encoder alone matches nouns/topics well but is
    close to blind on verb polarity (read_file vs write_file score nearly
    identically for "read" and "write" queries) - BM25 catches exactly
    that because it matches tokens literally. Fusing ranks instead of
    scores sidesteps the fact that cosine similarity and BM25 scores live
    on completely different, incomparable scales."""

    def __init__(self, model: str = "all-MiniLM-L6-v2"):
        self.vector = VectorIndex(model)
        self.bm25: BM25Okapi | None = None
        self.n = 0

    def build(self, entries: list):
        self.n = len(entries)
        self.vector.build([e.embed_text() for e in entries])
        self.bm25 = BM25Okapi([_tokenize(e.lexical_text()) for e in entries])

    def search(self, query: str, top_k: int | None = None) -> list[tuple[int, float]]:
        """Returns the fused ranking, best first. top_k=None returns every
        entry ranked (used by search_tools so it can apply a per-server cap
        over the full ranking rather than a pre-truncated top-K)."""
        vector_ranking = [i for i, _ in self.vector.search(query, top_k=self.n)]
        bm25_scores = self.bm25.get_scores(_tokenize(query))
        bm25_ranking = [int(i) for i in np.argsort(-bm25_scores, kind="stable")]
        fused = reciprocal_rank_fusion([vector_ranking, bm25_ranking])
        return fused[:top_k] if top_k is not None else fused