-- [P1-SUPERMARKET-NAME-NORMALIZE · 2026-07-26]
-- Normaliza `supermarket_products.master_food_name` al nombre CANÓNICO de `master_ingredients`.
--
-- POR QUÉ: la auditoría de variedad del 2026-07-26 midió que `supermarket_products` referencia 247
-- nombres de alimento mientras el catálogo tiene 204, y que 27 de esos nombres son VARIANTES que no
-- enlazan (singular/plural o nombre parcial). Eso deja ~300 productos sin poder conectarse a su
-- alimento — justo el puente que el roadmap de `/supermercado` necesita para que el usuario elija
-- marca desde la lista de compras.
--
-- CURADO A MANO, NO DIFUSO. Un emparejamiento por similitud de texto sobre alimentos en español
-- produce enlaces sencillamente falsos: el mío, que era razonable, propuso `Pasta de tomate` →
-- `Salsa de tomate`, `Queso de oveja` → `Queso de hoja`, `Margarina` → `Mantequilla`,
-- `Harina de trigo integral` → `Pan integral familiar`, `Leche semidescremada` → `Leche descremada`,
-- `Cereza` → `Cereza maraschino` y `Durazno` → `Durazno en almíbar`. Son alimentos DISTINTOS y el
-- usuario compraría lo que no es. Esos SIETE quedan fuera a propósito y se listan al final.
--
-- IDEMPOTENTE: cada UPDATE filtra por el nombre viejo, así que re-correrla no hace nada. Sólo toca
-- filas cuyo destino EXISTE en el catálogo (subconsulta EXISTS) — si el alimento aún no está
-- insertado, la fila se queda como está en vez de apuntar al vacío.

BEGIN;

-- Plural / singular y nombre parcial → canónico del catálogo.
DO $$
DECLARE
    _pares text[][] := ARRAY[
        -- [viejo, canónico]
        ARRAY['Aceituna',            'Aceitunas'],
        ARRAY['Yogurt Griego',       'Yogurt griego entero'],
        ARRAY['Galleta de soda',     'Galletas de soda'],
        ARRAY['Guandules',           'Gandules'],
        ARRAY['Huevos',              'Huevo'],
        ARRAY['Camarón',             'Camarones'],
        ARRAY['Orégano',             'Orégano dominicano'],
        ARRAY['Habichuela negra',    'Habichuelas negras'],
        ARRAY['Maíz dulce',          'Maíz dulce en granos'],
        ARRAY['Garbanzo',            'Garbanzos'],
        ARRAY['Fresa',               'Fresas'],
        ARRAY['Lenteja',             'Lentejas'],
        ARRAY['Longaniza',           'Longaniza dominicana'],
        ARRAY['Espinaca',            'Espinacas'],
        ARRAY['Tofu',                'Tofu firme'],
        ARRAY['Cúrcuma molida',      'Cúrcuma'],
        ARRAY['Algas marinas',       'Nori'],
        ARRAY['Filete Arenque',      'Arenque'],
        ARRAY['Kale Picado',         'Kale'],
        ARRAY['Galleta de soda integral', 'Galletas de soda'],
        ARRAY['Sandia',              'Sandía'],
        ARRAY['Guisantes secos',     'Guisantes secos']   -- ya canónico; presente para el conteo
    ];
    _i int;
    _n int;
    _total int := 0;
BEGIN
    FOR _i IN 1 .. array_length(_pares, 1) LOOP
        UPDATE public.supermarket_products s
           SET master_food_name = _pares[_i][2]
         WHERE s.master_food_name = _pares[_i][1]
           AND _pares[_i][1] <> _pares[_i][2]
           AND EXISTS (SELECT 1 FROM public.master_ingredients m
                        WHERE m.name = _pares[_i][2]);
        GET DIAGNOSTICS _n = ROW_COUNT;
        _total := _total + _n;
        IF _n > 0 THEN
            RAISE NOTICE '[P1-SUPERMARKET-NAME-NORMALIZE] % -> % : % producto(s)',
                _pares[_i][1], _pares[_i][2], _n;
        END IF;
    END LOOP;
    RAISE NOTICE '[P1-SUPERMARKET-NAME-NORMALIZE] total normalizado: %', _total;
END $$;

-- Sanity: ningún master_food_name debe apuntar a un nombre viejo ya normalizado.
DO $$
DECLARE _quedan int;
BEGIN
    SELECT COUNT(*) INTO _quedan
      FROM public.supermarket_products
     WHERE master_food_name IN ('Aceituna', 'Guandules', 'Huevos', 'Camarón', 'Garbanzo',
                                'Lenteja', 'Fresa', 'Espinaca', 'Habichuela negra', 'Kale Picado');
    IF _quedan > 0 THEN
        RAISE EXCEPTION '[P1-SUPERMARKET-NAME-NORMALIZE] quedaron % filas con nombre viejo', _quedan;
    END IF;
END $$;

COMMIT;

-- ─────────────────────────────────────────────────────────────────────────────────────────────────
-- EXCLUIDOS A PROPÓSITO (son alimentos DISTINTOS, no variantes de nombre). Si alguien los "arregla"
-- después, el usuario compraría otra cosa:
--
--   Pasta de tomate            ≠ Salsa de tomate        (concentrado vs salsa; 11 productos)
--   Queso de oveja             ≠ Queso de hoja          (se añade como alimento propio)
--   Margarina                  ≠ Mantequilla            (grasa vegetal vs láctea)
--   Harina de trigo integral   ≠ Pan integral familiar  (harina vs pan horneado)
--   Leche semidescremada       ≠ Leche descremada       (distinta grasa; 7 productos)
--   Cereza                     ≠ Cereza maraschino      (fresca vs en sirope)
--   Durazno                    ≠ Durazno en almíbar     (fresco vs en sirope)
--
-- Cada uno necesita su propia fila en `master_ingredients` con macros de fuente (USDA) y precio RD,
-- no un alias. Quedan como deuda documentada.
