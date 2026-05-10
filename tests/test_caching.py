import os
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import inspect

# Check if cached_content is in kwargs or signature
sig = inspect.signature(ChatGoogleGenerativeAI)
print("ChatGoogleGenerativeAI parameters:", list(sig.parameters.keys()))
if hasattr(ChatGoogleGenerativeAI, "cached_content"):
    print("Supports cached_content natively via attribute!")
else:
    print("No cached_content attribute directly.")
