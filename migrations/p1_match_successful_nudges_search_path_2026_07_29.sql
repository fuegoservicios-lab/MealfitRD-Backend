-- [P1-NUDGE-RPC-SEARCH-PATH · 2026-07-29]
-- `match_successful_nudges` fallaba el 100% de las veces desde la migración a Neon.
--
-- SÍNTOMA (log de producción, cron `run_proactive_checks`, recurrente):
--     proactive_agent: Error en embedding de nudge:
--     operator does not exist: extensions.vector <=> extensions.vector
--
-- CAUSA (SELECT forense contra Neon, 2026-07-29):
--     extensión `vector`                  -> schema `extensions`
--     match_successful_nudges  proconfig  -> {search_path=public, pg_catalog}   <-- SIN extensions
--     match_user_facts         proconfig  -> {search_path=public, extensions}   <-- hermana SANA
-- El TIPO resuelve (está cualificado como `extensions.vector`) pero el OPERADOR `<=>` no: los
-- operadores no se cualifican, se buscan por search_path. De las 7 RPC vectoriales del sistema,
-- esta era la única sin `extensions`; el barrido `fix_match_similar_plan_search_path_extensions`
-- arregló a una hermana y se saltó a esta porque 5 de las 7 no tienen SSOT en migrations/.
--
-- IMPACTO: fail-open. `proactive_agent` traga la excepción y deja `proven_strategies_text = ""`,
-- así que el nudge SE ENVÍA igual, pero sin las estrategias que ya funcionaron con usuarios en
-- situación similar. O sea: el bloque "Embedding-based Nudge Personalization" llevaba ~12 semanas
-- 100% apagado sin que ninguna alerta lo dijera (el cron reporta éxito porque no aborta).
--
-- SEGUNDO DEFECTO, en el mismo cuerpo: el filtro de sentimiento está en ESPAÑOL
-- ('motivado','aliviado','agradecido','determinado') mientras TODO el código escribe y lee valores
-- en INGLÉS — cron_tasks.py:17065-17066 ('frustration','sadness','guilt','annoyed' /
-- 'motivation','positive','curiosity'), :17453 ('motivation','positive','happy','excited') y
-- :17744. En la tabla viva el único valor presente es 'neutral'. Con el search_path arreglado, ese
-- IN(...) no evaluaría true JAMÁS y el único disyuntor operativo sería `meal_logged = true`,
-- reduciendo a la mitad el reclutamiento del corpus de nudges exitosos. Se alinea al vocabulario de
-- cron_tasks.py:17453, que es la query hermana con la misma intención ("nudges que funcionaron").
--
-- Idempotente: CREATE OR REPLACE + sanity DO $$ que falla ruidoso si el search_path no quedó bien.
-- SSOT: este archivo vive en `migrations/` Y en `backend/migrations/` (P3-MIGRATIONS-SSOT).

-- ⚠️ El TIPO va cualificado como `extensions.vector` en la firma a propósito: al aplicar esta
-- migración, la sesión que la ejecuta NO tiene por qué llevar `extensions` en su search_path — y de
-- hecho el primer intento falló con `type vector does not exist`. Una migración que arregla un
-- problema de search_path no puede depender del search_path de quien la aplica.

CREATE OR REPLACE FUNCTION public.match_successful_nudges(
    query_embedding extensions.vector,
    match_threshold double precision,
    match_count integer
)
RETURNS TABLE(
    id integer,
    nudge_content text,
    response_sentiment character varying,
    similarity double precision
)
LANGUAGE plpgsql
-- `extensions` añadido: es lo único que hacía fallar al operador `<=>`.
SET search_path TO 'public', 'extensions', 'pg_catalog'
AS $function$
BEGIN
    RETURN QUERY
    SELECT
        n.id,
        n.nudge_content,
        n.response_sentiment,
        1 - (n.context_embedding <=> query_embedding) as similarity
    FROM nudge_outcomes n
    WHERE n.context_embedding IS NOT NULL
      AND n.nudge_content IS NOT NULL
      -- vocabulario alineado al que el clasificador ESCRIBE (inglés); antes era español y no
      -- matcheaba nunca. `meal_logged` sigue siendo el disyuntor fuerte.
      AND (n.meal_logged = true
           OR n.response_sentiment IN ('motivation', 'positive', 'happy', 'excited'))
      AND 1 - (n.context_embedding <=> query_embedding) > match_threshold
    ORDER BY n.context_embedding <=> query_embedding
    LIMIT match_count;
END;
$function$;

-- Sanity: si el search_path no quedó con `extensions`, la RPC seguiría rota en silencio.
DO $$
DECLARE
    _cfg text[];
BEGIN
    SELECT p.proconfig INTO _cfg
    FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
    WHERE p.proname = 'match_successful_nudges' AND n.nspname = 'public'
    LIMIT 1;

    IF _cfg IS NULL OR NOT EXISTS (
        SELECT 1 FROM unnest(_cfg) AS c WHERE c LIKE '%extensions%'
    ) THEN
        RAISE EXCEPTION
            '[P1-NUDGE-RPC-SEARCH-PATH] match_successful_nudges quedo SIN extensions en su '
            'search_path (proconfig=%). El operador <=> volveria a fallar y el fallo es '
            'fail-open: nadie se enteraria.', _cfg;
    END IF;
END $$;

COMMENT ON FUNCTION public.match_successful_nudges(extensions.vector, double precision, integer) IS
    '[P1-NUDGE-RPC-SEARCH-PATH · 2026-07-29] RAG de nudges exitosos. `extensions` DEBE estar en el '
    'search_path: el operador <=> de pgvector vive ahi y no se cualifica. Sin el, la RPC falla el '
    '100% de las veces y proactive_agent lo traga (fail-open) -> personalizacion apagada en '
    'silencio. El filtro de response_sentiment usa el vocabulario INGLES del clasificador.';
