import httpx
import asyncio
import os
import sys

async def test():
    key = "sk_ed6a9f1ea3ae48cabd2e44df9be5fc2d9c470580bed397af"
    voice_id = "EXAVITQu4vr4xnSDxMaL"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": key
    }
    payload = {
        "text": "Probando",
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(url, json=payload, headers=headers)
        print("Status Code:", resp.status_code)
        if resp.status_code != 200:
            print("Error:", resp.text)
        else:
            print("Success! Got audio of length", len(resp.content))

if __name__ == "__main__":
    asyncio.run(test())
