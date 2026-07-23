import json

import tiktoken

from embeddings import VectorIndex

ENC = tiktoken.get_encoding("cl100k_base")
TOP_K = 5

# What the client actually sees with the router in place — fixed, always
# these two, regardless of how big the real catalog gets.
ROUTER_TOOL_SCHEMAS = [
    {"name": "search_tools",
     "description": "Search all available downstream tools by describing "
                    "what you want to do. Returns the top matching schemas.",
     "inputSchema": {"type": "object",
                     "properties": {"query": {"type": "string"}},
                     "required": ["query"]}},
    {"name": "invoke_tool",
     "description": "Invoke a downstream tool by its namespaced name with "
                    "a JSON object of arguments.",
     "inputSchema": {"type": "object",
                     "properties": {"tool_name": {"type": "string"},
                                    "arguments": {"type": "object"}},
                     "required": ["tool_name", "arguments"]}},
]


def tokens(obj) -> int:
    """How many tokens would this cost if sent as part of a request."""
    return len(ENC.encode(json.dumps(obj)))


def main():
    with open("catalog.json") as f:
        catalog = json.load(f)
    with open("benchmark_queries.json") as f:
        queries = json.load(f)

    baseline_tokens = tokens(catalog)
    router_fixed_tokens = tokens(ROUTER_TOOL_SCHEMAS)

    # Rebuild the same index the real server uses, so search behaves
    # identically to what you tested by hand in Inspector.
    index = VectorIndex()
    index.build([f"{c['name'].split('__', 1)[-1]}: {c['description']}"
                 for c in catalog])

    hits = 0
    per_query_tokens = []
    print(f"{'query':45} {'hit?':6} {'tokens (router)':16}")
    print("-" * 70)

    for q in queries:
        ranked = index.search(q["query"], top_k=TOP_K)
        retrieved = [catalog[i] for i, _ in ranked]
        retrieved_names = [r["name"] for r in retrieved]

        hit = q["expected"] in retrieved_names
        hits += hit

        query_tokens = router_fixed_tokens + tokens(retrieved)
        per_query_tokens.append(query_tokens)

        print(f"{q['query']:45} {'YES' if hit else 'NO':6} {query_tokens:16}")

    avg_router_tokens = sum(per_query_tokens) / len(per_query_tokens)

    print("\n--- Summary ---")
    print(f"Tools in real catalog:            {len(catalog)}")
    print(f"Baseline (all tools, every turn):  {baseline_tokens} tokens")
    print(f"Router (avg tokens per lookup):     {avg_router_tokens:.0f} tokens")
    print(f"Reduction on lookup turns:          "
          f"{100 * (1 - avg_router_tokens / baseline_tokens):.1f}%")
    print(f"Retrieval accuracy (hit@{TOP_K}):        "
          f"{hits}/{len(queries)} ({100 * hits / len(queries):.0f}%)")


if __name__ == "__main__":
    main()