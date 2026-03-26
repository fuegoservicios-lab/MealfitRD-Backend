import os
from dotenv import load_dotenv

load_dotenv()
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "")

from agent import generate_chat_title_background

if __name__ == "__main__":
    generate_chat_title_background("test_session_123", "Hola, quiero saber más sobre mi dieta y opciones C.")
