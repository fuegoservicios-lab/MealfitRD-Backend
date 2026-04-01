import os
import sys

from agent import chat_with_agent_stream

def main():
    print("Empezando el test de chat síncrono...")
    session_id = "test_session_123"
    prompt = "hola, qué tal?"
    
    for chunk in chat_with_agent_stream(session_id, prompt):
        print(chunk, end="")
            
if __name__ == "__main__":
    main()
