import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SEP = "__"


def _win_wrap(command: str, args: list[str]) -> tuple[str, list[str]]:
    """On Windows, .cmd shims (npx, npm, etc.) can't be spawned directly.
    Wrap them through cmd.exe, same fix that unblocked Phase 2."""
    if sys.platform == "win32" and command in ("npx", "npm", "uvx"):
        return "cmd", ["/c", command] + args
    return command, args


@dataclass
class CatalogEntry:
    server: str          # which downstream server this came from, e.g. "filesystem"
    name: str             # original tool name on that server, e.g. "read_file"
    namespaced: str       # "filesystem__read_file" — what the outside world sees
    description: str
    input_schema: dict

    def embed_text(self) -> str:
        """Text fed to the embedding model. Truncated to roughly one
        sentence: descriptions vary wildly in length across servers (up to
        65x in practice), and embedding the full text put long, verbose
        tools on a different similarity scale than terse ones - this keeps
        every entry comparable."""
        desc = self.description.strip()
        cutoff = desc.find(". ")
        if cutoff != -1 and cutoff < 200:
            desc = desc[:cutoff + 1]
        else:
            desc = desc[:200]
        return f"{self.name}: {desc}"

    def lexical_text(self) -> str:
        """Text fed to the BM25 index. Unlike embed_text, kept full length -
        BM25 already normalizes by document length, and tool-name tokens
        (read_file vs write_file, git_diff vs git_commit) are exactly the
        signal embeddings blur together."""
        return f"{self.name} {self.description}"

    def schema_json(self) -> dict:
        return {
            "name": self.namespaced,
            "description": self.description,
            "inputSchema": self.input_schema,
        }


@dataclass
class DownstreamPool:
    sessions: dict = field(default_factory=dict)    # server name -> live ClientSession
    entries: list = field(default_factory=list)     # every tool, from every server
    _stack: AsyncExitStack = field(default_factory=AsyncExitStack)

    async def connect_all(self, servers_config: dict):
        """servers_config looks like:
        {"filesystem": {"command": "npx", "args": [...]}, "memory": {...}}
        Connects to every server in the dict, one after another."""
        for name, spec in servers_config.items():
            command, args = _win_wrap(spec["command"], spec.get("args", []))
            params = StdioServerParameters(command=command, args=args,
                                            env=spec.get("env"))

            # enter_async_context keeps each connection open for the pool's
            # whole lifetime, and makes sure they all get cleaned up together
            # when close() is called (or if something errors out mid-setup).
            read, write = await self._stack.enter_async_context(stdio_client(params))
            session = await self._stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            self.sessions[name] = session

            tools = await session.list_tools()
            for t in tools.tools:
                self.entries.append(CatalogEntry(
                    server=name,
                    name=t.name,
                    namespaced=f"{name}{SEP}{t.name}",
                    description=t.description or "",
                    input_schema=t.inputSchema or {},
                ))
            print(f"  connected to '{name}': {len(tools.tools)} tools", file=sys.stderr)

    async def call(self, namespaced_name: str, arguments: dict):
        """Given 'filesystem__read_file', figure out which server owns it,
        strip the prefix, and forward the call there."""
        server, _, real_name = namespaced_name.partition(SEP)
        if server not in self.sessions:
            raise ValueError(f"No connected server named '{server}'")
        return await self.sessions[server].call_tool(real_name, arguments)

    async def close(self):
        await self._stack.aclose()