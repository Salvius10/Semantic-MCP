import asyncio
from catalog import DownstreamPool
from config import load_server_config
from embeddings import VectorIndex

async def main():
    pool = DownstreamPool()
    await pool.connect_all(load_server_config())

    index = VectorIndex()
    index.build([e.embed_text() for e in pool.entries])

    queries = [
        "read the contents of a file",
        "remember something for later",
        "list what's in a folder",
    ]
    for q in queries:
        print(f"\nQuery: {q!r}")
        for i, score in index.search(q, top_k=3):
            entry = pool.entries[i]
            print(f"  {score:.3f}  {entry.namespaced}")

    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())