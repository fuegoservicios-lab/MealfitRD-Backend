import os
import sys

# Asegurar que los imports funcionen si se ejecuta desde backend/scripts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_core import execute_sql_write, execute_sql_query

def migrate_semantic_cache():
    print("Iniciando migración de Semantic Cache para la tabla meal_plans...")
    
    # 1. Añadir columna vector
    try:
        query_add_col = "ALTER TABLE meal_plans ADD COLUMN IF NOT EXISTS profile_embedding vector(768);"
        execute_sql_write(query_add_col)
        print("✅ Columna profile_embedding vector(768) añadida o ya existente.")
    except Exception as e:
        print(f"❌ Error añadiendo columna profile_embedding: {e}")
        return

    # 2. Añadir índice HNSW (más rápido para búsqueda de similitud coseno que IVFFLAT)
    try:
        # HNSW index for cosine distance
        query_add_idx = "CREATE INDEX IF NOT EXISTS meal_plans_profile_emb_idx ON meal_plans USING hnsw (profile_embedding vector_cosine_ops);"
        execute_sql_write(query_add_idx)
        print("✅ Índice HNSW vector_cosine_ops añadido.")
    except Exception as e:
        print(f"⚠️ Error añadiendo índice HNSW (puede que falten datos o pgvector no soporte HNSW, no es bloqueante): {e}")

    # 3. Crear función RPC match_similar_plan
    # Esta función busca el plan más similar donde el usuario coincida o la similitud sea muy alta
    # No necesariamente filtramos por user_id, el objetivo es encontrar cualquier plan similar (incluso de otro usuario) 
    # y adaptarlo. Pero si queremos ser conservadores, podemos no filtrar por user_id para reutilizar el conocimiento global.
    try:
        query_rpc = """
        CREATE OR REPLACE FUNCTION match_similar_plan (
            query_embedding vector(768),
            match_threshold float,
            match_count int
        )
        RETURNS TABLE (
            id uuid,
            user_id uuid,
            plan_data jsonb,
            similarity float
        )
        LANGUAGE sql
        AS $$
            SELECT
                id,
                user_id,
                plan_data,
                1 - (meal_plans.profile_embedding <=> query_embedding) as similarity
            FROM meal_plans
            WHERE 1 - (meal_plans.profile_embedding <=> query_embedding) > match_threshold
            ORDER BY meal_plans.profile_embedding <=> query_embedding
            LIMIT match_count;
        $$;
        """
        execute_sql_write(query_rpc)
        print("✅ Función RPC match_similar_plan creada o actualizada.")
    except Exception as e:
        print(f"❌ Error creando RPC match_similar_plan: {e}")

if __name__ == "__main__":
    migrate_semantic_cache()
    print("Migración completada.")
