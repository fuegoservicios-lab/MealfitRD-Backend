import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def test():
    print("Testing gemini-3.1-flash-lite-preview...")
    try:
        start = time.time()
        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.1)
        res = llm.invoke("Di 'hola'")
        end = time.time()
        print(f"Success in {end - start:.2f}s: {res.content}")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test()
