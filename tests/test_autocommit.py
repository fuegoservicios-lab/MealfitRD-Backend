import asyncio
from db_core import async_connection_pool

async def main():
    await async_connection_pool.open()
    async with async_connection_pool.connection() as conn:
        print(f"Connection autocommit is: {conn.autocommit}")
    await async_connection_pool.close()

asyncio.run(main())
