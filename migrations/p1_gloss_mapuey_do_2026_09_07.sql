-- [P1-GLOSS-MAPUEY-DO · 2026-09-07]
-- El gloss del mapuey, que va en la dirección CONTRARIA a los 22 de agosto.
--
-- Los glosses de `p1_country_gloss_es_2026_08_23.sql` explican un dominicanismo a un extranjero
-- («Auyama (calabaza)», «Chinola (maracuyá)»): el nombre canónico es el que un dominicano
-- reconoce y el paréntesis es para los demás. Por eso la regla de render está cerrada para DO.
--
-- El mapuey es el caso inverso, reportado por el dueño: en RD la gente dice «ñame» y «mapuey»
-- suena a otra cosa. Aquí el nombre canónico es el RARO y el paréntesis es el que aclara.
--
-- POR QUÉ «ñame indio» Y NO «ñame» A SECAS. `Ñame` es OTRA FILA del catálogo, viva y con su
-- propio precio (RD$76/lb contra RD$99/lb del mapuey) y su propio fdc_id (170071 contra 169238):
-- son Dioscorea alata y Dioscorea trifida. Glosarlo como «ñame» mandaría a comprar el tubérculo
-- equivocado, RD$23/lb más barato. «Ñame indio» es uno de sus nombres reales en RD —junto a
-- «yampí» y «ñame blanco»— y dice que ES un ñame sin confundirlo con EL ñame.
--
-- Y NO «ñame morado»: la pulpa del mapuey es CLARA. Esa era mi primera propuesta y la corrigió
-- la fuente que trajo el dueño antes de que se escribiera nada.
--
-- RESTRICCIÓN DURA, la misma de agosto: `master_ingredients.name` es identidad canónica y NO se
-- toca. `gloss_es` es display-only — nunca entra en aliases ni en matching. Añadir «ñame indio»
-- como ALIAS sí sería peligroso: «ñame» es subcadena de «ñame indio» y colisionaría con la otra
-- fila (la clase de bug que este proyecto lleva 17 veces documentada: sal⊂salsa, res⊂fresco,
-- pollo⊂repollo). Como gloss, no toca ninguna clave de resolución.
--
-- Idempotente: re-aplicarla no cambia el resultado.

UPDATE public.master_ingredients
SET gloss_es = 'ñame indio'
WHERE name = 'Mapuey'
  AND gloss_es IS DISTINCT FROM 'ñame indio';

-- Sanity: la fila tiene que existir y quedar glosada, y `Ñame` tiene que seguir SIN gloss
-- (glosarla también sería la confusión que esta migración evita, pero al revés).
DO $$
DECLARE
    _mapuey TEXT;
    _name_gloss TEXT;
BEGIN
    SELECT gloss_es INTO _mapuey FROM public.master_ingredients WHERE name = 'Mapuey';
    IF _mapuey IS DISTINCT FROM 'ñame indio' THEN
        RAISE EXCEPTION 'P1-GLOSS-MAPUEY-DO: Mapuey no quedó glosado (gloss_es=%)', _mapuey;
    END IF;

    SELECT gloss_es INTO _name_gloss FROM public.master_ingredients WHERE name = 'Ñame';
    IF _name_gloss IS NOT NULL THEN
        RAISE EXCEPTION
            'P1-GLOSS-MAPUEY-DO: Ñame ganó un gloss (%) — son dos filas distintas y glosar las '
            'dos reintroduce la confusión que esta migración cierra', _name_gloss;
    END IF;
END $$;
