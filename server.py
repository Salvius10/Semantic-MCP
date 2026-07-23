# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp",
#     "sentence-transformers",
#     "numpy",
# ]
# ///
import asyncio
import json
import sys
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from catalog import DownstreamPool
from config import load_server_config
from embeddings import VectorIndex

TOP_K = 5

pool = DownstreamPool()
index: VectorIndex | None = None


def _build_index() -> VectorIndex:
    # Runs in a worker thread: importing sentence-transformers/torch and
    # encoding embeddings together take real time (seconds, sometimes tens
    # of seconds on a cold cache). Blocking the lifespan on this would delay
    # the MCP `initialize` response past clients' startup timeouts, so this
    # is kicked off as a background task instead - search_tools waits on it
    # if called before it finishes.
    idx = VectorIndex()
    idx.build([e.embed_text() for e in pool.entries])
    return idx


@asynccontextmanager
async def lifespan(_server):
    global index
    print("Discovering downstream servers...", file=sys.stderr)
    servers = load_server_config()
    if not servers:
        print("  no MCP servers found in Claude Desktop, Claude Code, Cursor, "
              "or Windsurf configs.", file=sys.stderr)
    await pool.connect_all(servers)
    print(f"Connected: {len(pool.entries)} tools across {len(pool.sessions)} "
          f"servers. Indexing in the background...", file=sys.stderr)

    async def index_when_ready():
        global index
        index = await asyncio.to_thread(_build_index)
        print(f"Ready: {len(pool.entries)} tools indexed.", file=sys.stderr)

    index_task = asyncio.create_task(index_when_ready())
    try:
        yield
    finally:
        index_task.cancel()
        await pool.close()


mcp = FastMCP("semantic-mcp-router", lifespan=lifespan)


@mcp.tool()
async def search_tools(query: str) -> str:
    """Search all available downstream tools by describing what you want to
    do. Returns the top matching tool schemas. Call this FIRST, then use
    invoke_tool with one of the returned names to actually run it."""
    if index is None:
        return json.dumps({
            "status": "indexing",
            "message": "Tool index is still building, try again in a few seconds."
        })
    hits = index.search(query, top_k=TOP_K)
    results = [
        {"score": round(score, 3), **pool.entries[i].schema_json()}
        for i, score in hits
    ]
    return json.dumps(results, indent=2)


@mcp.tool()
async def invoke_tool(tool_name: str, arguments: dict) -> str:
    """Invoke a downstream tool by its namespaced name (e.g.
    'filesystem__read_file'), as returned by search_tools, with a JSON
    object of arguments."""
    result = await pool.call(tool_name, arguments)
    parts = [
        block.text if getattr(block, "type", "") == "text" else str(block)
        for block in result.content
    ]
    return "\n".join(parts)


if __name__ == "__main__":
    mcp.run(transport="stdio")