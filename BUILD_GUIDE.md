# Build Guide: Semantic MCP Router

Learning-first build plan. Each phase has a goal, concepts you should
understand before/while writing the code, concrete tasks, and a checkpoint
that proves the phase actually works before you move on. Don't skip
checkpoints — this project lives or dies on "does the round-trip actually
work," and MCP's SDK surface has genuinely changed across versions, so
assumptions from training data are not reliable here.

Budget: ~30h total. Rough per-phase estimates included; your mileage varies
if this is your first MCP project.

---

## Phase 0: Environment setup (~1h)

**Goal:** a working Python environment and a way to poke at MCP servers
interactively before writing your own.

**Concepts:**
- What MCP (Model Context Protocol) actually is: a JSON-RPC-based protocol
  for a client (Claude Desktop, Cursor, the Inspector) to discover and call
  "tools" exposed by a server, over stdio or HTTP/SSE transport.
- `tools/list` and `tools/call` are the two RPC methods you care about most.

**Tasks:**
1. `pip install mcp openai numpy tiktoken` (or use `uv`, your call) in a
   fresh virtualenv. 
2. Install the **MCP Inspector** (`npx @modelcontextprotocol/inspector`) —
   this is your debugging tool for the whole project. You'll use it instead
   of Claude Desktop for every intermediate step because it shows you the
   raw JSON-RPC traffic.
3. Install one trivial downstream server to poke at — the reference
   `filesystem` server (`npx -y @modelcontextprotocol/server-filesystem
   <some-dir>`) is the standard "hello world" here.
4. Run the filesystem server through Inspector. Confirm you can see its
   `tools/list` output and call one tool (e.g. `read_file`) manually.

**Checkpoint:** you've seen a real `tools/list` response with real JSON
schemas in Inspector, and successfully invoked one tool. If you haven't
done this, everything downstream is guesswork.

---

## Phase 1: Spec refresher + minimal scaffolding (~2-3h)

**Goal:** your own MCP server exists, exposes exactly one dummy tool, and
round-trips through Inspector. This is the "does my SDK understanding even
work" checkpoint — do this before writing any real logic.

**Concepts:**
- `FastMCP` (the high-level decorator-based server API in the Python SDK)
  vs. the low-level `Server` class. Confirm which one current docs
  recommend — the scaffold brief flags this as unverified.
- Tool registration: how a Python function becomes an MCP tool (decorator,
  docstring → description, type hints → JSON schema).
- Lifespan/startup hooks — you'll need this later to connect to downstream
  servers when your router starts up, not per-request.

**Tasks:**
1. Look up the current `python-sdk` README/examples on PyPI or GitHub
   directly — don't trust remembered API shapes. Specifically confirm:
   - How `FastMCP(...)` is constructed, and whether `lifespan=` is still
     the mechanism for startup/shutdown resources.
   - The decorator name for registering a tool (`@mcp.tool()` as of recent
     versions).
2. Write `src/router/server.py` with a single dummy tool, e.g.
   `ping() -> str` returning `"pong"`.
3. Run it under stdio, connect via Inspector, call `ping`.

**Checkpoint:** Inspector shows one tool (`ping`), calling it returns
`"pong"`. Commit this. It's your known-good baseline for the SDK wiring.

---

## Phase 2: Connect to real downstream servers (~4-5h)

**Goal:** `src/router/catalog.py` connects to 2-3 real downstream MCP
servers as a *client*, pulls their `tools/list`, namespaces the tool names,
and dumps everything to `catalog.json`. This is also your "before" number
for the benchmark.

**Concepts:**
- Your router is simultaneously an MCP **server** (to the client) and an
  MCP **client** (to each downstream server). Keep these mental models
  separate — `catalog.py` is all client-side code.
- `stdio_client` + `ClientSession`: the pattern for spawning a subprocess
  MCP server and handshaking with it (`initialize` → `tools/list`).
- Namespacing collisions: two servers might both have a tool called
  `search`. Your `server__tool` convention solves this — but decide now
  whether `__` could collide with a legitimate tool/server name containing
  double underscore, and note that as a known limitation if so.

**Tasks:**
1. Pick 2-3 downstream servers that need no auth: `filesystem`, `memory`
   are the brief's suggestions. (Optional 3rd: `everything` reference
   server, good for testing edge cases like unusual schemas.)
2. Write a `config.example.json` describing how to launch each (command +
   args), matching whatever shape you decide `catalog.py` reads.
3. In `catalog.py`: for each configured server, spawn it via
   `stdio_client`, open a `ClientSession`, call `initialize()`, then
   `list_tools()`.
4. Build the namespaced catalog: `{"filesystem__read_file": {...schema},
   "memory__create_entity": {...schema}, ...}`.
5. Write `index.py` as the one-shot script: load config → build catalog →
   dump `catalog.json` to disk.
6. Run it. Open `catalog.json` and actually read it — this is real data
   you'll use for the benchmark's "before" token count and for
   `queries.json`'s expected-tool answers.

**Checkpoint:** `catalog.json` exists, has real tool names/descriptions/
schemas from at least 2 servers, correctly namespaced, no collisions. Note
the total token count of this file (tiktoken it) — that's your baseline
"tools sent on every request" number.

---

## Phase 3: Embeddings + cosine search, standalone (~3-4h)

**Goal:** given a query string and the catalog, return the top-K most
relevant tools — as a pure function you can test in a script or REPL,
with no MCP plumbing involved yet.

**Concepts:**
- Embedding a short text (tool name + description, maybe + param names) →
  a fixed-length vector. OpenAI's `text-embedding-3-small` is the obvious
  default (cheap, fast, good enough at this scale).
- Cosine similarity as a single matmul: normalize all tool embedding
  vectors once, normalize the query vector, dot product → similarity
  scores. This is the "no vector DB needed" trick — at low hundreds of
  tools this is microseconds in numpy.
- Caching: you don't want to re-embed the same catalog every run. Cache
  keyed on... think about what should invalidate the cache (tool
  description text changing? catalog.json's mtime/hash?).

**Tasks:**
1. Write `embeddings.py`:
   - A function to embed a batch of texts (the catalog entries) →
     `np.ndarray` of shape `(N, dim)`.
   - Cache these embeddings to a JSON (or `.npy`) file keyed by some hash
     of the tool's description text, so re-running doesn't re-call the
     API for unchanged tools.
   - A `search(query: str, k: int) -> list[str]` function: embed the
     query, cosine-similarity against the cached matrix, return top-K
     namespaced tool names + scores.
2. Test this standalone against `catalog.json` from Phase 2, with queries
   you make up by hand (e.g. "read a file from disk" should surface
   `filesystem__read_file` near the top).
3. Decide what "top-K" returns to the caller — just names, or names +
   scores + full schema? (You'll need full schema eventually since
   `search_tools`'s result IS the tool description the calling model uses
   to decide whether/how to call `invoke_tool`.)

**Checkpoint:** running `search("read a file")` against your real catalog
returns `filesystem__read_file` in the top 3, with no MCP server running —
just a script. This isolates embedding bugs from MCP wiring bugs, which
matters a lot for debugging Phase 4.

---

## Phase 4: Wire `search_tools` / `invoke_tool` end-to-end (~6-8h)

**Goal:** the actual router. A client connecting via Inspector sees
exactly 2 tools. Calling `search_tools` returns real candidates from your
catalog. Calling `invoke_tool` actually forwards to and gets a real result
back from the correct downstream server.

**Concepts:**
- Your server's lifespan now needs to: on startup, connect to all
  downstream servers (Phase 2's logic) and load/build the embedding cache
  (Phase 3's logic) — do this once, not per-call.
- `search_tools(query: str, k: int = 5) -> ...`: what should the *return
  type* be? It needs to give the calling model enough to construct a
  correct `invoke_tool` call — i.e., the tool's name, description, and
  full JSON schema for its arguments. Returning just names defeats the
  purpose.
- `invoke_tool(name: str, arguments: dict)`: strip the `server__` prefix,
  look up which live downstream session owns that server, forward
  `tools/call` with `arguments`, return the result unchanged (per the
  brief: no transformation, no summarization).
- Error handling worth thinking through (not gold-plating, just the
  obvious failure modes): unknown tool name passed to `invoke_tool`,
  downstream server process died, downstream call itself errors.

**Tasks:**
1. In `server.py`, wire lifespan startup: connect to all downstream
   servers from config, keep sessions alive for the process lifetime,
   build/load the embedding cache from the resulting catalog.
2. Implement `search_tools` using Phase 3's `search()`.
3. Implement `invoke_tool`: parse namespace, route to the right
   `ClientSession`, call, return result.
4. Test via Inspector first (per the brief — don't jump straight to
   Claude Desktop): connect, confirm you see exactly 2 tools, call
   `search_tools("create a memory entity")`, then take a returned tool
   name and manually call `invoke_tool` with it.
5. Only after Inspector round-trips cleanly, wire it into Claude Desktop's
   config and test a real agentic turn.

**Checkpoint:** in Claude Desktop, ask it to do something that requires a
downstream tool (e.g. "read this file" or "remember X"). Confirm in the
transcript/logs that it called `search_tools` first, then `invoke_tool`
with a namespaced name, and got a correct result. This is the demo-GIF
moment.

---

## Phase 5: Benchmark (~4-5h)

**Goal:** real numbers — token reduction and retrieval accuracy — not
vibes.

**Concepts:**
- Baseline token count: what a client would send on every request with
  the full tool list attached, using `tiktoken` on the concatenated
  schemas from `catalog.json`.
- Router token count: 2 tool definitions (`search_tools`, `invoke_tool`)
  plus, per query, whatever `search_tools` actually returns. Report this
  as a per-turn number since it varies by query.
- hit@K: for each test query with a known correct expected tool, does the
  correct tool appear in the top-K results? Report hit@3, hit@5, hit@10 —
  the brief specifically wants you to show where returns diminish, not
  just one cherry-picked K.

**Tasks:**
1. Replace `queries.json`'s placeholder expected-tool names with real
   namespaced names pulled from your actual `catalog.json` (the brief
   flags this explicitly — placeholders will silently produce meaningless
   accuracy numbers).
2. Write realistic queries per tool — phrase them the way an LLM
   mid-agentic-loop would, not like keyword search (e.g. "I need to save
   this fact for later" rather than "memory create_entity").
3. In `run_benchmark.py`: compute baseline tokens (full catalog) vs.
   router tokens (2 tools + typical search result), and hit@3/5/10 across
   all queries.
4. Actually look at the misses. Which queries retrieve the wrong tool?
   Is it an ambiguous query, a bad tool description, or a real embedding
   limitation? This qualitative bit is what makes the README credible
   instead of just a percentage.

**Checkpoint:** a table of baseline vs. router tokens, hit@3/5/10
percentages, and a short list of concrete misses with your read on why
they missed.

---

## Phase 6: README + polish (~4-5h)

**Goal:** a README that reads like an engineer wrote it, not like an AI
wrote a resume bullet.

**Tasks:**
1. Architecture diagram (the one in `project-brief.md` is a fine start).
2. Benchmark results as a chart/table from Phase 5's real numbers.
3. The differentiation section from the brief (mcp-gauge, GitHub's
   `--dynamic-toolsets`, IBM ContextForge) — write these comparisons
   yourself in your own words once you've actually built the thing; you
   understand the tradeoffs concretely now.
4. **Known Limitations section** — port the three items from the brief
   (large tool responses pass through unchanged; prompt-caching
   interaction — report both raw tokens and note it doesn't map 1:1 to
   dollar savings; no telemetry yet). Being upfront about these is a
   credibility signal, not a weakness — reviewers notice when a project
   overclaims.
5. 30-second demo GIF of the Phase 4 checkpoint (Claude Desktop calling
   `search_tools` → `invoke_tool`).
6. Optional stretch (only if time remains): the telemetry idea stolen
   from mcp-gauge — log every `invoke_tool` call with the query that
   triggered its retrieval, even just to a local file. Small addition,
   directly closes the gap you identified against a competitor.

**Checkpoint:** you could hand this repo to a stranger and they'd
understand what it does, why it exists, what it doesn't solve, and see
real numbers backing the claims.

---

## Ground rules while building

- Verify SDK API shapes against current docs before writing code that
  depends on them — this scaffold was written without network access and
  says so explicitly. Don't propagate that uncertainty into your own code
  without checking.
- Test through MCP Inspector before Claude Desktop at every phase from
  Phase 1 onward. Inspector shows raw protocol traffic; Claude Desktop
  hides failures behind "the model didn't call the tool" ambiguity.
- Don't scope-creep into multi-transport (SSE/HTTP) support, a real vector
  DB, or a full telemetry dashboard. All three are explicitly out of scope
  for v1 per the brief — they're good "future work" bullets, not v1 tasks.
- If you get stuck on a phase for way longer than its estimate, that's
  useful signal about which part of MCP/embeddings you actually don't
  understand yet — worth pausing to read docs rather than pushing through
  by trial and error.
