from db import get_user_chat_sessions
import json

sessions = get_user_chat_sessions("32569d67-f3c2-43c5-8467-86260efcafa0")
print(json.dumps(sessions, indent=2))
