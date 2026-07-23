def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Combine multiple rank-ordered lists of entry indices (best first)
    into one fused ranking. Each ranker only needs to contribute an
    ordering, not a comparable score - which is what makes RRF a good fit
    for combining cosine similarity (bounded, dense) with BM25 (unbounded,
    sparse): they'd never agree on scale, but rank position is comparable.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])
