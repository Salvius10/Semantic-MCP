import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from catalog import _win_wrap
from config import load_server_config

async def main():
    servers = load_server_config()
    name, spec = next(iter(servers.items()))
    command, args = _win_wrap(spec["command"], spec.get("args", []))
    params = StdioServerParameters(command=command, args=args, env=spec.get("env"))
    print(f"Testing raw stdio connection to '{name}'...")
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            for tool in result.tools:
                print(f"- {tool.name}: {tool.description}")
  
if __name__ == "__main__":
    asyncio.run(main())