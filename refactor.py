import sys
import re

with open('cron_tasks.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Extract _process_user
process_match = re.search(r'    def _process_user\(user\):\n(.+?)(?=\n    # MEJORA 5)', text, re.DOTALL)
process_body = process_match.group(1)

lines = process_body.split('\n')
dedented_lines = []
for line in lines:
    if line.startswith('            '):
        dedented_lines.append(line[8:])
    elif line.startswith('        '):
        dedented_lines.append(line[4:])
    else:
        dedented_lines.append(line)

new_process_func = 'def _process_user(user):\n' + '\n'.join(dedented_lines) + '\n'

new_run_func = """
def enqueue_nightly_rotations():
    logger.info("🕒 [CRON] Enqueuing Nightly Auto-Rotation for Premium Users...")
    query = '''
        SELECT id FROM user_profiles 
        WHERE (plan_tier IN ('plus', 'ultra', 'admin') OR subscription_status = 'ACTIVE')
        AND (health_profile->>'autoRotateMeals')::boolean = true
    '''
    try:
        users = execute_sql_query(query, fetch_all=True)
    except Exception as e:
        logger.error(f"❌ [CRON] Error querying auto-rotation users: {e}")
        return
        
    if not users:
        return
        
    logger.info(f"🔄 [CRON] Found {len(users)} users. Enqueuing...")
    for u in users:
        user_id = u['id']
        try:
            execute_sql_write('''
                INSERT INTO nightly_rotation_queue (user_id, status)
                SELECT %s, 'pending'
                WHERE NOT EXISTS (
                    SELECT 1 FROM nightly_rotation_queue WHERE user_id = %s AND status = 'pending'
                )
            ''', (user_id, user_id))
        except Exception as e:
            logger.warning(f"⚠️ [CRON] Failed to enqueue user {user_id}: {e}")

def run_nightly_auto_rotation():
    # Backward compat
    enqueue_nightly_rotations()

def process_rotation_queue():
    logger.info("🕒 [CRON] Checking nightly rotation queue...")
    query = '''
        SELECT id, user_id FROM nightly_rotation_queue
        WHERE status = 'pending'
        ORDER BY created_at ASC
        LIMIT 5;
    '''
    try:
        pending_tasks = execute_sql_query(query, fetch_all=True)
    except Exception as e:
        logger.error(f"❌ [CRON] Error querying rotation queue: {e}")
        return

    if not pending_tasks:
        return
        
    logger.info(f"🔄 [CRON] Processing {len(pending_tasks)} users from the queue.")
    
    for t in pending_tasks:
        execute_sql_write("UPDATE nightly_rotation_queue SET status = 'processing' WHERE id = %s", (t['id'],))
        
    for task in pending_tasks:
        user_id = str(task['user_id'])
        user_query = "SELECT id, health_profile FROM user_profiles WHERE id = %s"
        try:
            u_data = execute_sql_query(user_query, (user_id,), fetch_all=False)
            if u_data:
                _process_user(u_data)
                execute_sql_write("UPDATE nightly_rotation_queue SET status = 'completed' WHERE id = %s", (task['id'],))
            else:
                execute_sql_write("UPDATE nightly_rotation_queue SET status = 'failed' WHERE id = %s", (task['id'],))
        except Exception as e:
            logger.error(f"❌ [CRON] Worker failed for user {user_id}: {e}")
            execute_sql_write("UPDATE nightly_rotation_queue SET status = 'failed' WHERE id = %s", (task['id'],))
"""

top_part = text[:text.find('def run_nightly_auto_rotation():')]
final_content = top_part + new_process_func + new_run_func

with open('cron_tasks.py', 'w', encoding='utf-8') as f:
    f.write(final_content)

print('File refactored successfully.')
