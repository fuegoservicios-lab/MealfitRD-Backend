-- [P2-WHITE-FISH-ALIAS-SPLIT · 2026-09-02] Los alias «mero» y «tilapia» vivían en DOS filas:
-- la genérica «Filete de pescado blanco» y las filas por especie «Mero» / «Tilapia». El índice
-- de alias desempata por orden de fila y la genérica ganaba: una receta con «filete de mero»
-- compraba «Paquete 32 Oz» genérico (RD$255) en vez de Mero por libra (RD$290) o Tilapia por
-- libra (RD$130). Medido 3 veces el 02-sep. Un alias debe resolver a UNA fila.
-- Idempotente: sólo quita los dos alias de la genérica si siguen ahí; «chillo» se queda (no tiene fila).
UPDATE public.master_ingredients
   SET aliases = array_remove(array_remove(aliases, 'mero'), 'tilapia')
 WHERE name = 'Filete de pescado blanco'
   AND ('mero' = ANY(aliases) OR 'tilapia' = ANY(aliases));

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Filete de pescado blanco' AND ('mero' = ANY(aliases) OR 'tilapia' = ANY(aliases))) THEN
        RAISE EXCEPTION 'P2-WHITE-FISH-ALIAS-SPLIT: la fila genérica sigue con alias de especie';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Mero' AND 'mero' = ANY(aliases)) THEN
        RAISE EXCEPTION 'P2-WHITE-FISH-ALIAS-SPLIT: la fila Mero perdió su alias';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Tilapia' AND 'tilapia' = ANY(aliases)) THEN
        RAISE EXCEPTION 'P2-WHITE-FISH-ALIAS-SPLIT: la fila Tilapia perdió su alias';
    END IF;
END $$;
