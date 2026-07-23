import asyncio
from catalog import DownstreamPool
from config import load_server_config

async def main():
    pool = DownstreamPool()
    print("Connecting...")
    await pool.connect_all(load_server_config())

    print(f"\nTotal merged tools: {len(pool.entries)}")
    for e in pool.entries:
        print(f"  {e.namespaced}")

    await pool.close()

if __name__ == "__main__":
    asyncio.run(main())