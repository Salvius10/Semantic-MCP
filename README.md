# Semantic MCP

A semantic router that sits between an MCP client (Claude Desktop, Claude
Code, Cursor, Windsurf, ...) and any number of downstream MCP servers.

Instead of the client loading every tool from every downstream server on
every turn, Semantic MCP exposes just **two tools** — `search_tools` and
`invoke_tool` — and uses local embeddings to find the right downstream tool
for a natural-language query on demand. This cuts the token cost of large
tool catalogs and keeps the client's tool list small no matter how many
servers you have connected.

## What it does

- **Discovers your MCP servers automatically.** On startup it reads the
  `mcpServers` config from Claude Desktop, Cursor, and Windsurf (whichever
  are installed) and merges them — no manual list to maintain. Add or
  remove a server in any of those clients and Semantic MCP picks it up the
  next time it starts.
- **Connects to all of them** over stdio and collects their full tool
  catalogs (`catalog.py`).
- **Embeds every tool's name + description** locally with
  `sentence-transformers` (`all-MiniLM-L6-v2`) and builds a similarity
  index (`embeddings.py`) — no external API calls, fully offline after the
  first model download.
- **Exposes two tools to the outer client:**
  - `search_tools(query)` — semantic search over every downstream tool,
    returns the top matches with their full schemas.
  - `invoke_tool(tool_name, arguments)` — calls the actual downstream tool
    (namespaced as `server__tool`, e.g. `filesystem__read_file`) and
    returns its result.

## Project layout

| File | Role |
|---|---|
| `server.py` | Entry point. The MCP router itself — exposes `search_tools`/`invoke_tool`. |
| `config.py` | Discovers downstream servers from Claude Desktop / Cursor / Windsurf configs. |
| `catalog.py` | Connects to downstream servers over stdio, namespaces and routes tool calls. |
| `embeddings.py` | Local embedding index used for semantic tool search. |

## Requirements

- [`uv`](https://docs.astral.sh/uv/) — that's it. `server.py` declares its
  own dependencies inline, so `uv run` creates an isolated environment and
  installs everything automatically the first time you run it.
- At least one MCP client installed with servers configured (Claude
  Desktop, Cursor, or Windsurf) — this is where Semantic MCP discovers
  downstream servers from.

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
Ready: 21 tools across 2 servers.
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

> **Note:** if you register Semantic MCP inside the *same* config file it
> reads for discovery, always set `SEMANTIC_MCP_SELF_NAME` to match the
> entry's own key. Without it, the router would see its own entry and try
> to spawn itself as a downstream server.

Restart the client after editing its config so it picks up the new server.

## How discovery works

`config.py` checks these locations, in order, and merges everything it
finds (first match wins on a name conflict):

1. Claude Desktop — `%APPDATA%\Claude\claude_desktop_config.json`
2. Cursor — `~/.cursor/mcp.json`
3. Windsurf — `~/.codeium/windsurf/mcp_config.json`

No config file of its own, no manual sync step — whatever is currently
configured in any of those clients is what gets used, every time the
router starts.
