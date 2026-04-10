import httpx
import asyncio

async def main():
    api_key = 'sk_f49d2d81190ff4279bc924a45f1296a1c0334b378852e0a7'
    url = "https://api.elevenlabs.io/v1/text-to-speech/EXAVITQu4vr4xnSDxMaL"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": api_key
    }
    payload = {
        "text": "Hello world",
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75
        }
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
            print(r.status_code)
            print(r.text)
        except Exception as e:
            print("ERROR", str(e))

asyncio.run(main())
