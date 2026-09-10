-- [P1-QUESO-DE-PAPA-A-GOUDA · 2026-09-10]
-- El gouda tenia valores de gouda con el puntero USDA del FETA.
--
-- De donde sale: el dueno pregunto si el «queso de papa» podia ser cualquier queso. La fila
-- «Queso de papa» resulto ser EDAM (USDA 173419, «Cheese, edam») con un nombre PUERTORRIQUENO, sin
-- precio dominicano; el gouda es practicamente el mismo queso y su supermercado vende 17. Los 8
-- platos dominicanos pasan a gouda (eso va en los JSON del backend, no aqui). Antes de mandar 8
-- platos a la fila del gouda se comprobo su procedencia contra la API de USDA:
--
--   Queso gouda  fdc_id 173420  ->  USDA dice «Cheese, FETA» (265 kcal, 14,2 g prot, 1.140 mg Na)
--   sus valores: 356 kcal, 24,9 g prot, 27,4 g grasa, 819 mg Na = «Cheese, gouda», fdc 171241
--
-- Mismo defecto que el tofu en P1-CATALOGO-DICE-QUE-PRODUCTO: el VALOR esta bien y el PUNTERO
-- miente. Se corrige el puntero; ningun numero cambia.
--
-- Idempotente: re-aplicarla no cambia el resultado.

UPDATE public.master_ingredients
SET fdc_id = 171241
WHERE name = 'Queso gouda' AND fdc_id IS DISTINCT FROM 171241;

-- Sanity: la fila existe, apunta al gouda y conserva sus valores (si no, alguien cambio el dato
-- entre la verificacion y la aplicacion, y hay que volver a mirar antes de seguir).
DO $$
DECLARE
    _n INT;
BEGIN
    SELECT count(*) INTO _n FROM public.master_ingredients
    WHERE name = 'Queso gouda' AND fdc_id = 171241
      AND kcal_per_100g = 356 AND protein_g_per_100g = 24.9 AND sodium_mg_per_100g = 819;
    IF _n <> 1 THEN
        RAISE EXCEPTION '[P1-QUESO-DE-PAPA-A-GOUDA] el gouda no quedo como se verifico (filas: %)', _n;
    END IF;
END $$;
