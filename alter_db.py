import os
import psycopg

def get_db_url():
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('SUPABASE_DB_URL='):
                return line.strip().split('=', 1)[1].strip("'").strip('"')
    return None

url = get_db_url()
if url:
    clean_url = url
    if '.supabase.' in clean_url and ':5432' in clean_url:
        clean_url = clean_url.replace(':5432', ':6543')
    with psycopg.connect(clean_url) as conn:
        with conn.cursor() as cur:

            try:
                cur.execute('ALTER TABLE nudge_outcomes ADD COLUMN response_sentiment VARCHAR(50);')
                conn.commit()
                print('Added response_sentiment column.')
            except Exception as e:
                print('Already exists or error:', e)
            
            try:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS abandoned_meal_reasons (
                        id SERIAL PRIMARY KEY,
                        user_id UUID,
                        meal_type VARCHAR(50),
                        reason VARCHAR(50),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                ''')
                conn.commit()
                print('Table abandoned_meal_reasons verified/created.')
            except Exception as e:
                print('Error creating abandoned_meal_reasons:', e)
