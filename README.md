# Semantic MCP Router

A hybrid-search router that sits between an MCP client (Claude Desktop,
Claude Code, Cursor, Windsurf, ...) and any number of downstream MCP
servers.

Instead of the client loading every tool from every downstream server on
every turn, the router exposes just **two tools** — `search_tools` and
`invoke_tool` — and finds the right downstream tool for a natural-language
query on demand. This cuts the token cost of large tool catalogs and keeps
the client's tool list small no matter how many servers you have connected.

## What it does

- **Discovers your MCP servers automatically.** On startup it reads the
  `mcpServers` config from Claude Desktop, Claude Code, Cursor, and
  Windsurf (whichever are installed) and merges them — no manual list to
  maintain. Add or remove a server in any of those clients and the router
  picks it up the next time it starts. See [How discovery
  works](#how-discovery-works).
- **Connects to every discovered server** over stdio and collects their
  full tool catalogs (`catalog.py`).
- **Responds to the client immediately once connected**, without waiting
  for the (slower) search index to finish building. See [Startup
  behavior](#startup-behavior).
- **Ranks tools with hybrid search**, not embeddings alone. See [How
  search works](#how-search-works).
- **Exposes two tools to the outer client:**
  - `search_tools(query)` — hybrid search over every downstream tool,
    capped at 2 results per server so one large server can't crowd out the
    rest, returns the top matches with their full schemas.
  - `invoke_tool(tool_name, arguments)` — calls the actual downstream tool
    (namespaced as `server__tool`, e.g. `filesystem__read_file`) and
    returns its result.

## Project layout

| File | Role |
|---|---|
| `server.py` | Entry point. The MCP router itself — exposes `search_tools`/`invoke_tool`. |
| `config.py` | Discovers downstream servers from Claude Desktop / Claude Code / Cursor / Windsurf configs. |
| `catalog.py` | Connects to downstream servers over stdio, namespaces and routes tool calls. |
| `embeddings.py` | Hybrid (embedding + BM25) index used for tool search. |
| `ranking.py` | Reciprocal rank fusion — combines the embedding and BM25 rankings. |
| `test_queries.json` | Query set for evaluating retrieval quality (server- and tool-level). |

## Requirements

- [`uv`](https://docs.astral.sh/uv/) — that's it. `server.py` declares its
  own dependencies inline, so `uv run` creates an isolated environment and
  installs everything automatically the first time you run it.
- At least one MCP client installed with servers configured (Claude
  Desktop, Claude Code, Cursor, or Windsurf) — this is where the router
  discovers downstream servers from.

## Install & run — one command

```bash
git clone https://github.com/Salvius10/Semantic-MCP.git
cd Semantic-MCP
uv run server.py
```

No venv to create, no `pip install` step. The first run installs
dependencies and downloads the `all-MiniLM-L6-v2` embedding model
(~80 MB) — that first run can take a few minutes; every run after that is
fast and fully offline.

This starts the router on stdio — it's meant to be launched by an MCP
client, not used standalone in a terminal. On startup you'll see it log
which servers it found and how many tools it indexed:

```
Discovering downstream servers...
  found 2 server(s) in Claude Desktop config: filesystem, memory
  connected to 'filesystem': 12 tools
  connected to 'memory': 9 tools
Connected: 21 tools across 2 servers. Indexing in the background...
Ready: 21 tools indexed.
```

## Connect it to a client

### Claude Code

```bash
claude mcp add -t stdio semantic-router -- uv run --directory "<repo>" server.py
```

### Claude Desktop, Cursor, Windsurf

Add an entry to that client's own MCP config file (e.g. Claude Desktop's
`%APPDATA%\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "semantic-router": {
      "command": "uv",
      "args": ["run", "--directory", "<repo>", "server.py"],
      "env": { "SEMANTIC_MCP_SELF_NAME": "semantic-router" }
    }
  }
}
```

> **Note:** if you register the router inside the *same* config file it
> reads for discovery, always set `SEMANTIC_MCP_SELF_NAME` to match the
> entry's own key. Without it, the router would see its own entry and try
> to spawn itself as a downstream server.

Restart the client after editing its config so it picks up the new server.

## How discovery works

`config.py` checks these locations, in order, and merges everything it
finds (first match wins on a name conflict):

1. **Claude Desktop** — `%APPDATA%\Claude\claude_desktop_config.json`
2. **Claude Code** — `~/.claude.json` (top-level `mcpServers` key only —
   servers registered with `claude mcp add` at project/local scope, stored
   under `projects.<path>.mcpServers`, aren't covered yet)
3. **Cursor** — `~/.cursor/mcp.json`
4. **Windsurf** — `~/.codeium/windsurf/mcp_config.json`

No config file of its own, no manual sync step — whatever is currently
configured in any of those clients is what gets used, every time the
router starts.

## Startup behavior

Connecting to downstream servers is fast; building the search index is
not (importing `sentence-transformers`/`torch` and encoding every tool's
description can take anywhere from a few seconds to tens of seconds on a
cold cache). Blocking the MCP `initialize` handshake on that would risk
tripping a client's startup timeout (Claude Code's is 30s) — so the two
are decoupled:

1. `lifespan()` connects to every discovered server, then yields
   immediately. The client's `initialize` handshake completes as soon as
   this step is done, independent of index build time.
2. Indexing runs in a background thread (`asyncio.to_thread`) after that.
3. If `search_tools` is called in the brief window before indexing
   finishes, it returns `{"status": "indexing", ...}` instead of erroring.

## How search works

`search_tools` doesn't rank by embedding similarity alone — that alone
matches the *topic* of a query well but is close to blind on *polarity*
(`read_file` and `write_file` score nearly identically for a query about
"a file", regardless of whether it says "read" or "write"). The full
pipeline (`embeddings.py`, `ranking.py`):

1. **Vector ranking** — cosine similarity between the query and each
   tool's embedded `name: description` text (truncated to ~1
   sentence/200 chars, since raw description length varies up to 65x
   across servers and would otherwise put verbose and terse tools on
   different similarity scales).
2. **Lexical ranking** — a BM25 index over each tool's full, untruncated
   `name description` text. This is what catches the verb: `read_file`
   vs `write_file`, `git_diff` vs `git_commit` are trivially separable by
   token overlap even when a bi-encoder blurs them together. Tools with
   zero token overlap with the query are dropped from this ranking
   entirely rather than kept at an arbitrary tie-broken position.
3. **Fusion** — the top 15 candidates from each ranking are combined via
   [reciprocal rank fusion](https://en.wikipedia.org/wiki/Learning_to_rank#Reciprocal_rank_fusion)
   (`ranking.py`), which combines *rank position* rather than raw scores —
   cosine similarity and BM25 scores live on incomparable scales, but
   "how high did each ranker place this tool" is comparable.
4. **Diversity cap** — the fused ranking is then walked top-down, taking
   at most 2 hits from any single server and backfilling from the next-
   best candidate, so one large or generically-worded server can't fill
   the entire result page.

## Known limitations

- No confidence/abstention signal yet — a query with no good match still
  returns its best-effort top-K rather than indicating low confidence.
  Cosine similarity and BM25 scores aren't on a scale where a fixed
  global cutoff is meaningful; this needs a relative signal (e.g. margin
  over the catalog mean) calibrated against real query volume.
- Claude Code servers registered at project/local scope (not top-level in
  `~/.claude.json`) aren't discovered yet.
- No cross-encoder reranking — if verb/polarity confusion turns out to
  still be an issue in practice after the BM25 fusion, reranking the top
  candidates with a small cross-encoder is the next lever, at the cost of
  extra latency per search.
