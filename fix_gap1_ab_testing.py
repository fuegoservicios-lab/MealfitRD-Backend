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
    print(f"Conectando a la DB para aplicar mejora de A/B Testing...")
    with psycopg.connect(clean_url) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute('''
                    CREATE TABLE IF NOT EXISTS learning_experiments (
                        id SERIAL PRIMARY KEY,
                        user_id UUID,
                        strategy_applied VARCHAR(50),
                        outcome_quality_score FLOAT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );
                ''')
                conn.commit()
                print('✅ Tabla `learning_experiments` verificada/creada exitosamente.')
            except Exception as e:
                print('❌ Error creando learning_experiments:', e)
else:
    print("❌ No se encontró SUPABASE_DB_URL en .env")
