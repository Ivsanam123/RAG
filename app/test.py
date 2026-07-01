import asyncio
from utils import get_vector_store   # adjust import if needed

async def main():
    store = await get_vector_store()
    print("Vector store created:", store)

asyncio.run(main())