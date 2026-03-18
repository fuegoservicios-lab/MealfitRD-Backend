import asyncio
import os
import sys

from app import lifespan
from fastapi import FastAPI
from agent import achat_with_agent_stream

app = FastAPI(lifespan=lifespan)

async def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    async with lifespan(app):
        print("Empezando el test de chat asíncrono...")
        session_id = "test_session_123"
        prompt = "hola, qué tal?"
        
        async for chunk in achat_with_agent_stream(session_id, prompt):
            print(chunk, end="")
            
if __name__ == "__main__":
    asyncio.run(main())
