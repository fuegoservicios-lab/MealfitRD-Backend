import os
import sys
from dotenv import load_dotenv

# Asegurar que leemos el .env correcto
load_dotenv('.env')

sys.path.append(os.getcwd())
from db import supabase

# Vamos a buscar el ultimo mensaje de modelo
res = supabase.table('agent_messages').select('*').eq('role', 'model').order('created_at', desc=True).limit(1).execute()
if not res.data:
    print('No model messages found')
    sys.exit(0)

msg = res.data[0]
print(f"Found message: {msg['id']} for session: {msg['session_id']}")
print(f"Content starts with: {msg['content'][:50]}")

# Test strict equality update
update_res = supabase.table('agent_messages').update({'feedback': 'up'}).eq('session_id', msg['session_id']).eq('role', 'model').eq('content', msg['content']).execute()
print(f"Strict equality update returned: {len(update_res.data)} rows affected")

# Restore to null
supabase.table('agent_messages').update({'feedback': None}).eq('id', msg['id']).execute()

# Try with fuzzy match
short_content = msg['content'][:100] + '%' if len(msg['content']) > 100 else msg['content'] + '%'
update_res_fuzzy = supabase.table('agent_messages').update({'feedback': 'down'}).eq('session_id', msg['session_id']).eq('role', 'model').ilike('content', short_content).execute()
print(f"Fuzzy update returned: {len(update_res_fuzzy.data)} rows affected")

