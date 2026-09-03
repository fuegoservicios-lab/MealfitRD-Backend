-- [P1-COUNTRY-GLOSS-SOLO-INGLES · 2026-08-23]
-- Gloss panhispánico display-only para regionalismos del catálogo.
--
-- RESTRICCIÓN DURA: master_ingredients.name es identidad canónica y NO se toca.
-- gloss_es solo viaja a display_gloss_es; nunca participa en aliases ni matching.
-- Idempotente: la columna y cada valor pueden re-aplicarse sin cambiar el resultado.

ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS gloss_es TEXT;

COMMENT ON COLUMN public.master_ingredients.gloss_es IS
    'Término panhispánico display-only para nombres regionales; NULL si el nombre canónico ya es ampliamente reconocible.';

UPDATE public.master_ingredients AS m
SET gloss_es = v.gloss
FROM (VALUES
    ('Ají cubanela', 'pimiento italiano'),
    ('Auyama', 'calabaza'),
    ('Batata', 'boniato o camote'),
    ('Casabe', 'pan crujiente de yuca'),
    ('Casabe albahaca', 'pan crujiente de yuca con albahaca'),
    ('Chinola', 'maracuyá'),
    ('Gandules', 'guandú'),
    ('Guineo', 'banana'),
    ('Guineo verde', 'plátano macho verde'),
    ('Habichuelas blancas', 'frijoles blancos'),
    ('Habichuelas negras', 'frijoles negros'),
    ('Habichuelas rojas', 'frijoles rojos'),
    ('Lechosa', 'papaya'),
    ('Longaniza dominicana', 'embutido fresco dominicano'),
    ('Molondrones', 'okra'),
    ('Panapén', 'fruta del pan'),
    ('Queso de hoja', 'queso fresco hilado'),
    ('Recao', 'culantro'),
    ('Tayota', 'chayote'),
    ('Tocineta', 'beicon o tocino'),
    ('Vainitas', 'judías verdes'),
    ('Yautía', 'malanga')
) AS v(name, gloss)
WHERE m.name = v.name
  AND m.gloss_es IS DISTINCT FROM v.gloss;

DO $$
DECLARE
    _bad TEXT;
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name = 'master_ingredients'
          AND column_name = 'gloss_es'
          AND data_type = 'text'
    ) THEN
        RAISE EXCEPTION '[P1-COUNTRY-GLOSS-SOLO-INGLES] falta master_ingredients.gloss_es TEXT';
    END IF;

    SELECT string_agg(v.name, ', ' ORDER BY v.name)
    INTO _bad
    FROM (VALUES
        ('Ají cubanela', 'pimiento italiano'),
        ('Auyama', 'calabaza'),
        ('Batata', 'boniato o camote'),
        ('Casabe', 'pan crujiente de yuca'),
        ('Casabe albahaca', 'pan crujiente de yuca con albahaca'),
        ('Chinola', 'maracuyá'),
        ('Gandules', 'guandú'),
        ('Guineo', 'banana'),
        ('Guineo verde', 'plátano macho verde'),
        ('Habichuelas blancas', 'frijoles blancos'),
        ('Habichuelas negras', 'frijoles negros'),
        ('Habichuelas rojas', 'frijoles rojos'),
        ('Lechosa', 'papaya'),
        ('Longaniza dominicana', 'embutido fresco dominicano'),
        ('Molondrones', 'okra'),
        ('Panapén', 'fruta del pan'),
        ('Queso de hoja', 'queso fresco hilado'),
        ('Recao', 'culantro'),
        ('Tayota', 'chayote'),
        ('Tocineta', 'beicon o tocino'),
        ('Vainitas', 'judías verdes'),
        ('Yautía', 'malanga')
    ) AS v(name, gloss)
    LEFT JOIN public.master_ingredients AS m ON m.name = v.name
    WHERE m.name IS NULL OR m.gloss_es IS DISTINCT FROM v.gloss;

    IF _bad IS NOT NULL THEN
        RAISE EXCEPTION '[P1-COUNTRY-GLOSS-SOLO-INGLES] regionalismos ausentes o sin gloss correcto: %', _bad;
    END IF;
END $$;

