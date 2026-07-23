import json
import os
import sys

# Priority order: if the same server name shows up in more than one client's
# config, whichever client appears first here wins. Every client below uses
# the same {"mcpServers": {name: {command, args, env?}}} shape, so no format
# translation is needed.
CANDIDATE_CONFIGS = [
    ("Claude Desktop", lambda: _appdata_path("Claude", "claude_desktop_config.json")),
    ("Claude Code", lambda: _home_path(".claude.json")),
    ("Cursor", lambda: _home_path(".cursor", "mcp.json")),
    ("Windsurf", lambda: _home_path(".codeium", "windsurf", "mcp_config.json")),
]


def _appdata_path(*parts: str) -> str | None:
    appdata = os.environ.get("APPDATA")
    return os.path.join(appdata, *parts) if appdata else None


def _home_path(*parts: str) -> str:
    return os.path.join(os.path.expanduser("~"), *parts)


def _read_mcp_servers(path: str | None) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"  warning: couldn't read {path}: {e}", file=sys.stderr)
        return {}
    return data.get("mcpServers", {})


def load_server_config() -> dict:
    """Auto-discover MCP servers from every installed client's config file,
    live, on every call. No manual export/sync step: whatever is currently
    configured in Claude Desktop, Claude Code, Cursor, or Windsurf is what
    gets used.

    Skips its own entry (matched by the SEMANTIC_MCP_SELF_NAME env var, set
    on this process by whoever launched it) so the router never tries to
    spawn itself as a downstream server when it's registered inside one of
    the same config files it reads from."""
    self_name = os.environ.get("SEMANTIC_MCP_SELF_NAME")
    merged: dict = {}
    for client_name, path_fn in CANDIDATE_CONFIGS:
        servers = _read_mcp_servers(path_fn())
        servers.pop(self_name, None)
        new = {name: spec for name, spec in servers.items() if name not in merged}
        if new:
            print(f"  found {len(new)} server(s) in {client_name} config: "
                  f"{', '.join(new.keys())}", file=sys.stderr)
        merged.update(new)
    return merged
