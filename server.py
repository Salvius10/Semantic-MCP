# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp",
#     "sentence-transformers",
#     "numpy",
# ]
# ///
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


@asynccontextmanager
async def lifespan(_server):
    global index
    print("Discovering downstream servers...", file=sys.stderr)
    servers = load_server_config()
    if not servers:
        print("  no MCP servers found in Claude Desktop, Cursor, or Windsurf "
              "configs.", file=sys.stderr)
    await pool.connect_all(servers)
    index = VectorIndex()
    index.build([e.embed_text() for e in pool.entries])
    print(f"Ready: {len(pool.entries)} tools across {len(pool.sessions)} servers.", file=sys.stderr)
    try:
        yield
    finally:
        await pool.close()


mcp = FastMCP("semantic-mcp-router", lifespan=lifespan)


@mcp.tool()
async def search_tools(query: str) -> str:
    """Search all available downstream tools by describing what you want to
    do. Returns the top matching tool schemas. Call this FIRST, then use
    invoke_tool with one of the returned names to actually run it."""
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