from db_core import execute_sql_write, connection_pool
import uuid

if connection_pool and not getattr(connection_pool, '_opened', False):
    connection_pool.open()

user_id = str(uuid.uuid4())
try:
    execute_sql_write("INSERT INTO auth.users (id, aud, role, email) VALUES (%s, 'authenticated', 'authenticated', 'test@test.com')", (user_id,))
    print(f"Inserted into auth.users: {user_id}")
    execute_sql_write("INSERT INTO user_profiles (id) VALUES (%s)", (user_id,))
    print(f"Inserted into user_profiles: {user_id}")
except Exception as e:
    print(f"Error: {e}")
