drop function if exists match_user_facts(vector(768), float, int, uuid);
drop function if exists hybrid_search_user_facts(text, vector(768), int, uuid);

create or replace function match_user_facts (
  query_embedding vector(768),
  match_threshold float,
  match_count int,
  p_user_id uuid
) returns table (
  id uuid,
  fact text,
  metadata jsonb,
  similarity float
)
language sql
as $$
  select
    user_facts.id,
    user_facts.fact,
    user_facts.metadata,
    -- Time & Intensity-Weighted Retrieval: Similitud Vectorial * Factor de Decaimiento (Time-Decay) * Intensidad
    -- Los hechos más recientes retienen su fuerza al 100% (1.0), los antiguos decaen (decay time-weight).
    (1 - (user_facts.embedding <=> query_embedding)) * 
    exp(-(extract(epoch from (now() - user_facts.created_at)) / 86400.0) / 365.0) *
    (COALESCE((user_facts.metadata->>'intensity')::float, 3.0) / 5.0) as similarity
  from user_facts
  where user_facts.embedding <=> query_embedding < 1 - match_threshold
    and user_facts.user_id = p_user_id
    and user_facts.is_active = true
  order by user_facts.embedding <=> query_embedding
  limit match_count;
$$;

create or replace function hybrid_search_user_facts (
  query_text text,
  query_embedding vector(768),
  match_count int,
  p_user_id uuid
) returns table (
  id uuid,
  fact text,
  metadata jsonb,
  similarity float
)
language sql
as $$
  select
    user_facts.id,
    user_facts.fact,
    user_facts.metadata,
    -- Time-Weighted Retrieval en Búsqueda Híbrida
    (1 - (user_facts.embedding <=> query_embedding)) * 
    exp(-(extract(epoch from (now() - user_facts.created_at)) / 86400.0) / 365.0) *
    (COALESCE((user_facts.metadata->>'intensity')::float, 3.0) / 5.0) as similarity
  from user_facts
  where user_facts.embedding <=> query_embedding < 1 - 0.5
    and user_facts.user_id = p_user_id
    and user_facts.is_active = true
  order by user_facts.embedding <=> query_embedding
  limit match_count;
$$;
