-- [P1-LIBRARY-V3-SIN-SUSTITUTO · 2026-09-08] `Avena` no admitia 'tostar' en prep_methods,
-- y hay una plantilla del Dish Registry que se LLAMA «Yogur con frutas picadas y avena
-- TOSTADA». Dos fuentes de produccion contradiciendose: el registry pide el plato y la
-- metadata culinaria declara imposible su tecnica.
--
-- Consecuencia medida: la capa V1 de culinary_coherence (`_v1_verbo_alimento`) marca
--   "'tostar' no esta en prep_methods('hervir','ninguno'): Tuesta la avena en una sarten..."
-- Es un FALSO POSITIVO — tostar avena es tecnica corriente (granola, avena tostada) — y su
-- coste no es cosmetico: ensucia `_shopping_coherence_block_history` y, si V1 escalara
-- alguna vez a block, tumbaria un plato legitimo.
--
-- COMO APARECIO: no lo encontro una auditoria del catalogo. Lo encontro la prosa de la
-- biblioteca de recetas (P1-RECIPE-LIBRARY-DO) al pasar por el escaner de produccion: 138 de
-- 140 limpias, y una de las 2 sucias era esto. La biblioteca esta funcionando como banco de
-- pruebas del escaner contra prosa real, que es mas de lo que se le pedia.
--
-- ALCANCE DELIBERADAMENTE MINIMO. El scan del catalogo encontro otros 8 alimentos sin
-- 'tostar' que "suenan tostables": Quinoa, Arroz integral, Garbanzos, Granola, Cacao en
-- polvo, Leche de avena/almendras, Mantequilla de almendras. NO se tocan: de `Avena` hay
-- EVIDENCIA (una receta real disparo el falso positivo y una plantilla la nombra); de los
-- demas solo hay intuicion mia, y la leccion de P1-CULINARY-METADATA-BETA es que las
-- asignaciones se validan por simulacion, no por criterio. Anadir 'tostar' donde solo lo
-- SUPONGO es como el palillo se volvio un ritual.
--
-- `Leche de avena` se queda fuera a proposito: es fila distinta y «tuesta la leche de avena»
-- SI debe seguir disparando.
--
-- Idempotente: el UPDATE lleva `NOT ('tostar' = ANY(prep_methods))`, asi que re-ejecutar es
-- no-op; y no crea la fila si no existe (un catalogo sin `Avena` es otro problema).

UPDATE public.master_ingredients
   SET prep_methods = array_append(prep_methods, 'tostar')
 WHERE name = 'Avena'
   AND prep_methods IS NOT NULL
   AND NOT ('tostar' = ANY(prep_methods));

-- == Sanity: si la fila existe, tiene que haber quedado con 'tostar' ======================
DO $$
DECLARE
  v_existe  boolean;
  v_tostar  boolean;
BEGIN
  SELECT EXISTS (SELECT 1 FROM public.master_ingredients WHERE name = 'Avena')
    INTO v_existe;
  IF NOT v_existe THEN
    RAISE NOTICE '[P1-LIBRARY-V3-SIN-SUSTITUTO] no hay fila `Avena`: nada que migrar';
    RETURN;
  END IF;

  SELECT bool_and('tostar' = ANY(prep_methods))
    FROM public.master_ingredients
   WHERE name = 'Avena' AND prep_methods IS NOT NULL
    INTO v_tostar;

  -- OJO: `RAISE EXCEPTION` no acepta `||` en su formato — el mensaje va en UNA cadena, o con `%`.
  -- La primera version llevaba concatenacion y reventó al aplicarla contra Neon, DESPUES de que el
  -- gate la diera por buena: el test de migraciones parsea el TEXTO, no ejecuta el SQL. Un test
  -- parser-based no puede decir que una migracion corre.
  IF v_tostar IS FALSE THEN
    RAISE EXCEPTION '[P1-LIBRARY-V3-SIN-SUSTITUTO] Avena sigue sin tostar en prep_methods tras el UPDATE';
  END IF;
END
$$;
