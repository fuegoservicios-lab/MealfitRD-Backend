-- [P1-5 · 2026-05-10] CHECK constraint de formato sobre
-- `chunk_lesson_telemetry.event` — backstop runtime contra typos.
--
-- Causa raíz (audit 2026-05-10):
--   La columna `event` es `text` libre sin validación. El test
--   `test_p1_hist_audit_5_lesson_event_whitelist.py` ya enforza drift a
--   nivel de CI (parsea cron_tasks.py + clasifica cada event), pero esa
--   defensa solo cubre call sites con literales estáticos
--   (`event="lesson_synthesized_low_confidence"`). Modos no cubiertos:
--     - F-strings dinámicos: `event=f"lesson_{kind}"` con `kind` calculado
--       en runtime.
--     - Call sites en módulos no parseados (admin tools, scripts, REPL).
--     - Hot patches del operador via SQL editor directo.
--     - Typos en code-paths que jamás disparan tests (edge cases).
--   Cualquiera de esos podía persistir un event con whitespace, capital,
--   guión, o longitud anómala — pasaba al filtro de `/lessons-counts`
--   (`event = ANY(%s)`) silenciosamente (no match → invisible al usuario).
--
-- Por qué CHECK de formato (regex) y no de enum (lista exacta de valores):
--   Una constraint enum (`event IN ('a','b','c')`) forzaría una migración
--   cada vez que un developer añade un event nuevo válido — fricción alta.
--   El formato regex (`^[a-z][a-z0-9_]+$`) atrapa los modos de typo reales
--   (espacios, capitales, hyphens, prefix numérico, longitud excesiva) sin
--   acoplar el schema a la lista de valores. La validación semántica
--   (lección vs métrica mecánica) sigue en el meta-test parser-based
--   P1-HIST-AUDIT-5 que ya enforza CI.
--
-- Defensa-en-profundidad: combinado con el meta-test, hay dos gates:
--   - CI (parser): catches semantic drift entre code y whitelist.
--   - Runtime (CHECK): catches malformed values que el parser no ve.

BEGIN;

ALTER TABLE public.chunk_lesson_telemetry
    DROP CONSTRAINT IF EXISTS chunk_lesson_telemetry_event_format;

ALTER TABLE public.chunk_lesson_telemetry
    ADD CONSTRAINT chunk_lesson_telemetry_event_format
    CHECK (
        event ~ '^[a-z][a-z0-9_]+$'
        AND length(event) BETWEEN 1 AND 100
    );

COMMENT ON CONSTRAINT chunk_lesson_telemetry_event_format
    ON public.chunk_lesson_telemetry IS
    '[P1-5 · 2026-05-10] Backstop runtime contra typos en event. Formato '
    'canónico: minúsculas + dígitos + underscore, empieza por letra, '
    '1-100 caracteres. La validación semántica (lección vs métrica) la '
    'enforza tests/test_p1_hist_audit_5_lesson_event_whitelist.py (CI '
    'parser-based). Esta constraint es el último gate para call sites '
    'no parseados (f-strings dinámicos, admin tools, REPL).';

COMMIT;
