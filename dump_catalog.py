import asyncio
import json

from catalog import DownstreamPool
from config import load_server_config

async def main():
    pool = DownstreamPool()
    await pool.connect_all(load_server_config())

    catalog = [entry.schema_json() for entry in pool.entries]
    with open("catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"Dumped {len(catalog)} real tool schemas -> catalog.json")
    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())