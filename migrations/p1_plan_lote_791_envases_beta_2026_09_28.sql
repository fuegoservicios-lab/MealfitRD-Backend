-- [P1-PLAN-LOTE-791 · 2026-09-28] G58 (datos): las filas de catálogo-país que se venden en ENVASE
-- no tenían ningún dato de envase, y la lista las rotulaba a peso.
--
-- MEDIDO (replay determinista del agregador sobre una copia de `master_ingredients`, sin DB ni IA,
-- auditoría G58/G13 del 28-sep): de las 130 filas sin precio del registro de catálogo-país, las 64 que
-- se venden en envase (26 paquete, 10 botella, 6 lata, 5 frasco, 5 funda, 4 sobre, 4 envase, 3 pote,
-- 1 litro) perdían el envase en la lista: «1 sobre de Azafrán» → «1 lb de Azafrán», «1 lata de
-- Anchoas» → «1 lb». Sin `container_weight_g` el agregador cae al peso por defecto de la categoría
-- (450 g Despensa, 500 g Proteínas, 1000 g Lácteos), pensado para RD. Y en DO, Dátiles y Cúrcuma
-- tienen unidad de envase (paquete, frasco) sin peso de envase: el mismo hueco.
--
-- QUÉ SE LLENA: `market_container` (el mismo envase que la fila ya declara en `default_unit`; Champús,
-- que decía «litro», va en «botella», y Sofrito, que decía «paquete», va en «frasco»: todas sus muestras
-- con marca son frascos de 12 oz), `container_weight_g` y `available_sizes_g` (un solo tamaño).
-- NO se toca `market_packages`: cada entrada exige un precio y estos países no tienen precios propios.
-- NO se inventa envase para las 57 filas que se venden a peso ni para las 9 de mazo o unidad.
--
-- PESO POR UNIDAD DE LOS SEIS CHILES SECOS (bloque 2b, revisión ronda 2). Los planes mexicanos escriben
-- el chile por CONTEO («3 chiles anchos»). Sin `density_g_per_unit` el conteo no se convierte a gramos,
-- el tope de condimentos lo deja en 1 paquete y 21 anchos (~357 g) salían «1 paquete» de 85 g, sin nota.
-- Con el peso por unidad el conteo se vuelve gramos y la lista compra los paquetes que cubren la receta.
-- REGLA: la porción «1 pepper» de USDA SR Legacy cuando existe para ESE chile (ancho 17 g, pasilla 7 g,
-- las mismas fdc_id que ya usa su nutrición); si no, la mediana de los conteos minoristas publicados
-- (Mexican Please, Spices Inc, CooksInfo), en gramo entero, mínimo 1. Sólo llena el hueco
-- (`density_g_per_unit IS NULL`) de filas SIN precio; la procedencia se añade a `container_source`.
-- Efecto declarado: la nutrición de «N chiles X» también resuelve gramos (antes no se contaban).
--
-- FUENTE: Open Food Facts (https://world.openfoodfacts.org), base de datos bajo licencia ODbL — la
-- atribución es obligatoria: «© colaboradores de Open Food Facts, ODbL». Se consultó el campo
-- `quantity` de los productos de cada país y categoría (search.openfoodfacts.org, 28-sep), se
-- descartaron a mano los productos que no eran el alimento (salsas, snacks, multipacks, formatos de
-- hostelería). REGLA DEL TAMAÑO, una sola para las 66 filas: el tamaño real más cercano a la mediana,
-- salvo cuando a la mediana la arrastran formatos que no son el de referencia (granel, hostelería,
-- tarrinas de 32-64 oz, monodosis) y el envase minorista que más se repite (la moda) es otro: entonces
-- la moda. Así salen Azafrán, Anchoas, Mazapán, Arándanos rojos, Ensalada de macarrones, Sazonador para
-- tacos, Adobo y Sofrito. Un formato FAMILIAR que se vende en cualquier súper no es granel: la lata de
-- 28 oz de Frijoles horneados no arrastra nada y ahí gana la mediana (16 oz). PR usa US como sustituto DECLARADO
-- (OFF casi no tiene productos de PR con cantidad). Donde el alimento no tenía ninguna muestra, se usó
-- la mediana de su CATEGORÍA y la fila lo dice («PROXY»). La procedencia de cada fila queda en la
-- columna nueva `container_source`: la auditoría de procedencia del catálogo ya mostró lo que cuesta
-- que el origen de un dato viva sólo en un comentario.
--
-- EL BLOQUE BETA JAMÁS TOCA UNA FILA CON PRECIO: exige `price_per_lb = 0 AND price_per_unit = 0` y
-- `container_weight_g IS NULL`. LA EXCEPCIÓN, DECLARADA: el bloque 2 SÍ toca dos filas DO con precio,
-- Dátiles y Cúrcuma, por nombre y sólo en el hueco del envase (`container_weight_g IS NULL AND
-- market_packages IS NULL`), sin leer ni escribir ninguna columna de precio. Sus valores NO tienen
-- evidencia dominicana: Dátiles sale de OFF US (340 g; en DO hace comprar un paquete entero para los
-- ~48 g de una receta) y Cúrcuma, del frasco de especiero de EE. UU. (57 g), cuando el único dato
-- dominicano de OFF es un Badia de 16 oz (435,6 g). ESPERAN EL OK DEL DUEÑO CON CAPTURA de su súper
-- antes de aplicar esta migración. Si no se quiere tocar DO en esta pasada no basta con
-- borrar ese bloque: hay que quitar también sus dos nombres del sanity y aplazar la constraint, que
-- las acusaría (son las dos únicas filas DO con unidad de envase y sin peso).
--
-- EL ANCLA (patrón I8, `meal_plans_complete_requires_days`): CHECK `master_ingredients_envase_requires_weight`
-- — una fila con unidad de envase (en `market_container` o en `default_unit`) exige
-- `container_weight_g > 0`. Sin ella, la próxima alta de catálogo vuelve a nacer sin envase con los
-- tests en verde (son parser-based: ninguno mira el DATO). La lista de unidades de envase es la de
-- `shopping_calculator._CONTAINER_UNIT_ALIASES`, y `test_p1_plan_lote_791.py` ancla la paridad.
-- `COALESCE(container_weight_g, 0)`: un CHECK que evalúa a NULL PASA; sin el COALESCE, NULL no mordería.
--
-- Idempotente (P3-MIGRATION-IDEMPOTENCE-DOC): ADD COLUMN IF NOT EXISTS, UPDATE sólo donde el envase es
-- NULL, DROP CONSTRAINT IF EXISTS antes de ADD, sanity en bloques DO con RAISE EXCEPTION.
-- SSOT dual-dir (P3-MIGRATIONS-SSOT): vive en migrations/ Y backend/migrations/.
-- tooltip-anchor: P1-PLAN-LOTE-791 (test_p1_plan_lote_791.py)

ALTER TABLE public.master_ingredients
    ADD COLUMN IF NOT EXISTS container_source text;

COMMENT ON COLUMN public.master_ingredients.container_source IS
    '[P1-PLAN-LOTE-791] Procedencia de market_container/container_weight_g/available_sizes_g (fuente, n, mediana, decisión) y, en los seis chiles secos, de density_g_per_unit (tras « | »). NULL = sin procedencia registrada.';

-- ── 1. Las 64 filas beta envasadas (sin precio RD) ────────────────────────────────────────────────
UPDATE public.master_ingredients AS m
SET market_container   = v.envase,
    container_weight_g = v.gramos,
    available_sizes_g  = jsonb_build_array(v.gramos),
    container_source   = v.fuente
FROM (VALUES
    ('Azafrán'                      , 'sobre'   ,    0.4, 'ES', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF ES en:saffron n=23: moda 0,4 g (x6), mediana 0,5 g -> 0,4 g, el sobre de Carmencita/Pote/Froiz'),
    ('Alioli'                       , 'frasco'  ,    180, 'ES', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF ES en:aiolis n=28: mediana = moda 180 g'),
    ('Anchoas'                      , 'lata'    ,     50, 'ES', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF ES en:anchovy-fillets n=36: moda 50 g (x9, la lata RO-50); mediana 77,5 g arrastrada por tripacks y latas de hostelería -> 50 g'),
    ('Mazapán'                      , 'paquete' ,    200, 'ES', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF ES en:marzipan n=42: mediana 225 g, moda 200 g (x13) -> 200 g, tamaño real más cercano'),
    ('Membrillo dulce'              , 'paquete' ,    400, 'ES', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF ES en:quince-cheeses n=30: mediana = moda 400 g'),
    ('Turrón'                       , 'paquete' ,    200, 'ES', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF ES en:turron n=69: mediana = moda 200 g (la tableta)'),
    ('Nata'                         , 'botella' ,    200, 'ES', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF ES en:creams + "nata" n=31: mediana = moda 200 ml (brik)'),
    ('Aceite de achiote'            , 'botella' ,    280, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: sin aceite de achiote con cantidad en OFF (MX/US/mundo); PROXY aceites saborizados OFF MX n=1: 280 g. Confianza baja: revisar con captura'),
    ('Achiote'                      , 'sobre'   ,    125, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF MX: único achiote mexicano con cantidad, recado rojo La Popular 125 g (EAN 75045746); US/ES n=5 de 50 a 460 g (mediana 227 g) -> 125 g'),
    ('Chile ancho'                  , 'paquete' ,     85, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PROXY de categoría: chiles secos OFF US+MX n=15 (0 a 8 por chile), mediana 85 g (3 oz)'),
    ('Chile chipotle'               , 'paquete' ,     85, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PROXY de categoría: chiles secos OFF US+MX n=15 (0 a 8 por chile), mediana 85 g (3 oz)'),
    ('Chile de árbol'               , 'paquete' ,     85, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PROXY de categoría: chiles secos OFF US+MX n=15 (0 a 8 por chile), mediana 85 g (3 oz)'),
    ('Chile guajillo'               , 'paquete' ,     85, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PROXY de categoría: chiles secos OFF US+MX n=15 (0 a 8 por chile), mediana 85 g (3 oz)'),
    ('Chile mulato'                 , 'paquete' ,     85, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PROXY de categoría: chiles secos OFF US+MX n=15 (0 a 8 por chile), mediana 85 g (3 oz)'),
    ('Chile pasilla'                , 'paquete' ,     85, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PROXY de categoría: chiles secos OFF US+MX n=15 (0 a 8 por chile), mediana 85 g (3 oz)'),
    ('Chocolate de mesa'            , 'paquete' ,    630, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF MX n=2 (Ibarra 630 g y 1,08 kg), US n=4 (77 a 907 g) -> 630 g, la caja estándar de Ibarra'),
    ('Flor de Jamaica'              , 'paquete' ,    227, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US flor seca n=4 (MX sin cantidades; fuera un té de 26 g): mediana 170 g, moda 227 g (x2, 8 oz) -> 227 g'),
    ('Frijoles refritos'            , 'lata'    ,    430, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF MX en:refried-beans n=16: mediana = moda 430 g'),
    ('Huitlacoche'                  , 'lata'    ,    186, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: los 2 huitlacoches de OFF son de 907 g (formato hostelería); PROXY hongos y vegetales en lata OFF MX n=8 (Herdez, La Costeña): mediana 186 g'),
    ('Panela'                       , 'paquete' ,    227, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US piloncillo n=4 (170 a 227 g, moda 227 g); la "Panela" LALA 400 g de OFF MX es queso: excluida'),
    ('Tortilla de maíz'             , 'paquete' ,    680, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US+MX tortillas de maíz n=35 (sin nopal ni hostelería): mediana 680 g (24 oz)'),
    ('Crema mexicana'               , 'pote'    ,    450, 'MX', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF MX (Lala, Alpura) n=9: mediana 450 ml (Crema Lala)'),
    ('Arequipe'                     , 'pote'    ,    400, 'CO', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: sin arequipe con cantidad en OFF CO; PROXY dulce de leche OFF mundo n=20: mediana 400 g (el único colombiano, manjar blanco, 450 g)'),
    ('Natilla'                      , 'pote'    ,    300, 'CO', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF CO (Su Sabor, EAN 770) n=3: 300 g'),
    ('Suero costeño'                , 'botella' ,    200, 'CO', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF CO: suero pasteurizado Klaren''s 200 g (EAN 770), n=1; PROXY fermentados lácteos CO n=15, moda 150 a 200 g -> 200 g'),
    ('Champús'                      , 'botella' ,   1000, 'CO', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: sin champús en OFF; PROXY bebidas de fruta OFF CO n=6: mediana = moda 1 L (la unidad del catálogo ya era "litro")'),
    ('Aderezo ranch'                , 'botella' ,    454, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=10 (sin sobres individuales): mediana 454 g (16 oz)'),
    ('Jarabe de arce'               , 'botella' ,    354, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:maple-syrups n=7 (sin el polvo): mediana 354 ml (12 fl oz)'),
    ('Kétchup'                      , 'botella' ,    567, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:ketchup n=17: mediana = moda 567 g (20 oz)'),
    ('Salsa barbacoa'               , 'botella' ,    510, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:barbecue-sauces n=12: mediana = moda 510 g (18 oz)'),
    ('Salsa inglesa'                , 'botella' ,    296, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:worcestershire-sauces n=14: mediana 290 g, moda 296 ml (x5, 10 fl oz)'),
    ('Crema agria'                  , 'envase'  ,    454, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:sour-creams n=25: mediana = moda 454 g (16 oz)'),
    ('Crema mitad y mitad'          , 'envase'  ,    473, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=8: mediana 473 ml (pinta)'),
    ('Ensalada de macarrones'       , 'envase'  ,    454, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=20: mediana 553 g, moda 454 g (x6, el envase de 16 oz de charcutería) -> 454 g'),
    ('Suero de mantequilla'         , 'envase'  ,    946, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:buttermilks n=2: 946 ml (cuarto de galón). n bajo'),
    ('Chile en polvo'               , 'frasco'  ,     71, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:chili-powders n=5: mediana = moda 71 g (2,5 oz)'),
    ('Arándanos rojos'              , 'funda'   ,    340, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US arándano fresco n=20: mediana 255 g, moda 340 g (x6, la funda de 12 oz) -> 340 g'),
    ('Bolitas de papa'              , 'funda'   ,    907, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=18: mediana = moda 907 g (32 oz)'),
    ('Malvaviscos'                  , 'funda'   ,    284, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:marshmallows n=35: mediana = moda 284 g (10 oz)'),
    ('Papas ralladas'               , 'funda'   ,    794, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=12: mediana 797 g -> 794 g (28 oz), tamaño real más cercano'),
    ('Pretzels'                     , 'funda'   ,    340, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=23: mediana 340 g (12 oz)'),
    ('Chili con carne'              , 'lata'    ,    425, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=13: mediana = moda 425 g (15 oz)'),
    ('Frijoles horneados'           , 'lata'    ,    454, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=21: mediana 454 g (16 oz); moda 794 g (28 oz)'),
    ('Salsa de salchicha'           , 'lata'    ,    425, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US: country sausage gravy Great Value 425 g, n=1 (el otro era un pot pie congelado). n bajo'),
    ('Bagels'                       , 'paquete' ,    510, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=20: mediana 498 g entre 485 y 510 g -> 510 g (18 oz, x3)'),
    ('Galletas Graham'              , 'paquete' ,    408, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=15: mediana = moda 408 g (14,4 oz)'),
    ('Masa para pie'                , 'paquete' ,    425, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=16: mediana = moda 425 g (15 oz)'),
    ('Mezcla para panqueques'       , 'paquete' ,    794, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:pancake-mixes n=26: mediana 794 g (28 oz)'),
    ('Pan de maíz'                  , 'paquete' ,    454, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=19: mediana = moda 454 g'),
    ('Panecillos de mantequilla'    , 'paquete' ,    425, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US "dinner rolls" n=10: mediana 425 g (15 oz); "butter rolls" devolvía mantequilla en rollo: descartada'),
    ('Panecillos ingleses'          , 'paquete' ,    340, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:english-muffins n=29: mediana = moda 340 g (6 unidades)'),
    ('Pepperoni'                    , 'paquete' ,    170, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US "sliced pepperoni" n=19: mediana = moda 170 g (6 oz)'),
    ('Queso en hebras'              , 'paquete' ,    227, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=17: mediana = moda 227 g (8 oz)'),
    ('Sémola de maíz'               , 'paquete' ,    907, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=17: mediana 907 g (2 lb)'),
    ('Wafles'                       , 'paquete' ,    255, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US en:waffles n=30: mediana 255 g (9 oz)'),
    ('Sazonador para tacos'         , 'sobre'   ,     28, 'US', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF US n=22: moda 28 g (x9, el sobre de 1 oz); mediana 71 g arrastrada por frascos y bolsas de 6 a 24 oz -> 28 g'),
    ('Aceitunas rellenas'           , 'frasco'  ,    340, 'PR', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PR usa US declarado. OFF US n=4 (sin multipack ni hostelería): mediana 354 g -> 340 g (12 oz)'),
    ('Adobo'                        , 'frasco'  ,    227, 'PR', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PR usa US declarado. OFF US n=23: moda 227 g (x4, Goya 8 oz, la marca de referencia en PR); mediana 340 g arrastrada por los 2 lb de Badia y el de 30 oz -> 227 g'),
    ('Alcaparrado'                  , 'frasco'  ,     99, 'PR', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PR usa US declarado. Sin alcaparrado con cantidad en OFF; PROXY alcaparras OFF US n=4: mediana 108,5 g -> 99 g'),
    ('Especias para arroz con dulce', 'sobre'   ,     28, 'PR', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PR usa US declarado. Sin la mezcla en OFF; PROXY especias enteras de formato chico (clavo, canela) OFF US n=6: mediana 28 g'),
    ('Harina de yuca'               , 'paquete' ,    454, 'PR', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PR usa US declarado. OFF US harinas de yuca n=5: mediana 454 g (1 lb)'),
    ('Pique'                        , 'botella' ,    148, 'PR', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PR usa US declarado. OFF US en:hot-sauces n=8: mediana 151,5 g -> 148 ml (5 fl oz)'),
    ('Ron de cocina'                , 'botella' ,    750, 'PR', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PR usa US declarado. OFF US ron n=4 (50 ml, 750 ml, 1,75 L x2) -> 750 ml, la botella estándar. n bajo'),
    ('Sofrito'                      , 'frasco'  ,    340, 'PR', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: PR usa US declarado. OFF US sofrito y recaito boricuas n=27: moda 340 g (x9, el frasco de 12 oz de Goya, Iberia y Loisa); mediana 680 g arrastrada por las tarrinas de 32 y 64 oz -> 340 g, en frasco (la fila dice paquete)')
) AS v(name, envase, gramos, pais, fuente)
WHERE m.name = v.name
  AND m.container_weight_g IS NULL
  AND COALESCE(m.price_per_lb, 0) = 0
  AND COALESCE(m.price_per_unit, 0) = 0;

-- ── 2. DO: Dátiles y Cúrcuma (la excepción explícita; sólo el hueco del envase) ───────────────────
UPDATE public.master_ingredients AS m
SET market_container   = v.envase,
    container_weight_g = v.gramos,
    available_sizes_g  = jsonb_build_array(v.gramos),
    container_source   = v.fuente
FROM (VALUES
    ('Dátiles', 'paquete' ,    340, 'DO', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: DO sin datos en OFF: US declarado. OFF US en:dates n=35: mediana = moda 340 g (12 oz)'),
    ('Cúrcuma', 'frasco'  ,     57, 'DO', '[P1-PLAN-LOTE-791] Open Food Facts (ODbL), revisado a mano 2026-09-28: OFF DO: único producto dominicano con cantidad, Badia 16 oz (435,6 g), formato grande; OFF US n=33: moda global 454 g (x9, bolsas a granel sin marca), frasco de especiero 57 g (x7: Badia, McCormick, Trader Joe''s) -> 57 g, el frasco de 2 oz. La fila se usa en los seis países. Revisar con captura')
) AS v(name, envase, gramos, pais, fuente)
WHERE m.name = v.name
  AND m.container_weight_g IS NULL
  AND m.market_container IS NULL
  AND m.market_packages IS NULL;

-- ── 2b. Peso de UN chile seco (revisión ronda 2): el conteo se vuelve gramos ──────────────────────
UPDATE public.master_ingredients AS m
SET density_g_per_unit = u.g_por_unidad,
    container_source   = concat_ws(' | ', m.container_source, u.fuente)
FROM (VALUES
    ('Chile ancho'   , 17, '[P1-PLAN-LOTE-791] peso por unidad: USDA FoodData Central SR Legacy fdc 169396 "Peppers, ancho, dried", porción "1 pepper" = 17 g (la misma fdc_id de la fila; CooksInfo 17 g, Mexican Please ~19 g)'),
    ('Chile pasilla' ,  7, '[P1-PLAN-LOTE-791] peso por unidad: USDA FoodData Central SR Legacy fdc 168579 "Peppers, pasilla, dried", porción "1 pepper" = 7 g (la misma fdc_id de la fila; Mexican Please ~9,5 g)'),
    ('Chile guajillo',  9, '[P1-PLAN-LOTE-791] peso por unidad: sin porción USDA; mediana de los conteos minoristas publicados: Mexican Please 6 por 2 oz (9,4 g), Spices Inc 2 por oz (14,2 g), CooksInfo 10 por 40 g (4 g) -> 9 g'),
    ('Chile mulato'  , 11, '[P1-PLAN-LOTE-791] peso por unidad: sin porción USDA; mediana de los conteos minoristas publicados: Spices Inc 2-3 por oz (11,3 g), n=1 -> 11 g. Confianza baja (el ancho, el mismo poblano seco, pesa 17 g en USDA)'),
    ('Chile chipotle',  3, '[P1-PLAN-LOTE-791] peso por unidad: sin porción USDA; mediana de los conteos minoristas publicados del chipotle morita: Mexican Please 16 por 2 oz (3,5 g), Spices Inc 9 por oz (3,2 g) -> 3 g'),
    ('Chile de árbol',  1, '[P1-PLAN-LOTE-791] peso por unidad: sin porción USDA propia (la genérica "Peppers, hot chile, sun-dried", fdc 168570, da 0,5 g); mediana de los conteos minoristas publicados: Mexican Please 50 por 2 oz (1,1 g), Spices Inc 50 por oz (0,6 g) -> 1 g (gramo entero, mínimo 1)')
) AS u(name, g_por_unidad, fuente)
WHERE m.name = u.name
  AND m.density_g_per_unit IS NULL
  AND COALESCE(m.price_per_lb, 0) = 0
  AND COALESCE(m.price_per_unit, 0) = 0;

-- ── 3. Sanity acotado al lote ─────────────────────────────────────────────────────────────────────
DO $$
DECLARE
    _beta text[] := ARRAY['Azafrán', 'Alioli', 'Anchoas', 'Mazapán', 'Membrillo dulce', 'Turrón', 'Nata', 'Aceite de achiote', 'Achiote', 'Chile ancho', 'Chile chipotle', 'Chile de árbol', 'Chile guajillo', 'Chile mulato', 'Chile pasilla', 'Chocolate de mesa', 'Flor de Jamaica', 'Frijoles refritos', 'Huitlacoche', 'Panela', 'Tortilla de maíz', 'Crema mexicana', 'Arequipe', 'Natilla', 'Suero costeño', 'Champús', 'Aderezo ranch', 'Jarabe de arce', 'Kétchup', 'Salsa barbacoa', 'Salsa inglesa', 'Crema agria', 'Crema mitad y mitad', 'Ensalada de macarrones', 'Suero de mantequilla', 'Chile en polvo', 'Arándanos rojos', 'Bolitas de papa', 'Malvaviscos', 'Papas ralladas', 'Pretzels', 'Chili con carne', 'Frijoles horneados', 'Salsa de salchicha', 'Bagels', 'Galletas Graham', 'Masa para pie', 'Mezcla para panqueques', 'Pan de maíz', 'Panecillos de mantequilla', 'Panecillos ingleses', 'Pepperoni', 'Queso en hebras', 'Sémola de maíz', 'Wafles', 'Sazonador para tacos', 'Aceitunas rellenas', 'Adobo', 'Alcaparrado', 'Especias para arroz con dulce', 'Harina de yuca', 'Pique', 'Ron de cocina', 'Sofrito'];
    _do   text[] := ARRAY['Dátiles', 'Cúrcuma'];
    _chiles text[] := ARRAY['Chile ancho', 'Chile pasilla', 'Chile guajillo', 'Chile mulato', 'Chile chipotle', 'Chile de árbol'];
    _n int;
    _lista text;
BEGIN
    -- Un nombre del lote que no existe es una errata: la fila seguiría sin envase en silencio.
    SELECT count(*) INTO _n FROM public.master_ingredients WHERE name = ANY(_beta || _do);
    IF _n <> 66 THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-791] esperaba 66 filas del lote, hay %', _n;
    END IF;

    SELECT count(*), string_agg(name, ', ') INTO _n, _lista FROM public.master_ingredients
    WHERE name = ANY(_beta || _do)
      AND NOT (COALESCE(container_weight_g, 0) > 0 AND market_container IS NOT NULL);
    IF _n > 0 THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-791] % filas del lote siguen sin envase: %', _n, _lista;
    END IF;

    -- Rango de la clase: del sobre de azafrán (0,4 g) a la bolsa de 32 oz (907 g) y el litro.
    SELECT count(*), string_agg(name, ', ') INTO _n, _lista FROM public.master_ingredients
    WHERE name = ANY(_beta || _do) AND (container_weight_g < 0.1 OR container_weight_g > 1500);
    IF _n > 0 THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-791] envases fuera de [0,1 g, 1,5 kg]: %', _lista;
    END IF;

    -- Los seis chiles secos con su peso por unidad (bloque 2b): sin él, «3 chiles» vuelve a 1 paquete.
    SELECT count(*), string_agg(name, ', ') INTO _n, _lista FROM public.master_ingredients
    WHERE name = ANY(_chiles) AND NOT (COALESCE(density_g_per_unit, 0) > 0);
    IF _n > 0 THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-791] chiles secos sin peso por unidad: %', _lista;
    END IF;

    -- Ninguna fila con precio, salvo la excepción DO declarada, lleva la procedencia de este lote.
    SELECT count(*), string_agg(name, ', ') INTO _n, _lista FROM public.master_ingredients
    WHERE container_source LIKE '[P1-PLAN-LOTE-791]%'
      AND (COALESCE(price_per_lb, 0) > 0 OR COALESCE(price_per_unit, 0) > 0)
      AND NOT (name = ANY(_do));
    IF _n > 0 THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-791] el bloque beta tocó filas CON precio: %', _lista;
    END IF;

    -- Guarda previa de la constraint: no intentes crearla sobre datos que la violan (y di cuáles).
    SELECT count(*), string_agg(name, ', ') INTO _n, _lista FROM public.master_ingredients
    WHERE NOT (
        (market_container IS NULL OR COALESCE(container_weight_g, 0) > 0)
        AND (lower(btrim(COALESCE(default_unit, ''))) <> ALL (ARRAY[
        'bolsa', 'bolsas', 'bolsita', 'bolsitas', 'botella', 'botellas', 'botellita', 'botellitas',
        'caja', 'cajas', 'carton', 'cartones', 'cartones.', 'cartón', 'cartón.', 'envase',
        'envases', 'frasco', 'frascos', 'funda', 'fundas', 'fundita', 'funditas', 'galon',
        'galones', 'galón', 'jarra', 'jarras', 'lata', 'latas', 'paquete', 'paquetes',
        'pote', 'potes', 'pqte', 'pqtes', 'sobre', 'sobrecito', 'sobrecitos', 'sobres',
        'tarro', 'tarros', 'tetra', 'tetra-pak', 'tetrapak'
        ]::text[]) OR COALESCE(container_weight_g, 0) > 0)
    );
    IF _n > 0 THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-791] % filas con unidad de envase y sin container_weight_g: %', _n, _lista;
    END IF;
END $$;

-- ── 4. El ancla en el dato ────────────────────────────────────────────────────────────────────────
ALTER TABLE public.master_ingredients
    DROP CONSTRAINT IF EXISTS master_ingredients_envase_requires_weight;

ALTER TABLE public.master_ingredients
    ADD CONSTRAINT master_ingredients_envase_requires_weight
    CHECK (
        (market_container IS NULL OR COALESCE(container_weight_g, 0) > 0)
        AND (lower(btrim(COALESCE(default_unit, ''))) <> ALL (ARRAY[
        'bolsa', 'bolsas', 'bolsita', 'bolsitas', 'botella', 'botellas', 'botellita', 'botellitas',
        'caja', 'cajas', 'carton', 'cartones', 'cartones.', 'cartón', 'cartón.', 'envase',
        'envases', 'frasco', 'frascos', 'funda', 'fundas', 'fundita', 'funditas', 'galon',
        'galones', 'galón', 'jarra', 'jarras', 'lata', 'latas', 'paquete', 'paquetes',
        'pote', 'potes', 'pqte', 'pqtes', 'sobre', 'sobrecito', 'sobrecitos', 'sobres',
        'tarro', 'tarros', 'tetra', 'tetra-pak', 'tetrapak'
        ]::text[]) OR COALESCE(container_weight_g, 0) > 0)
    );

COMMENT ON CONSTRAINT master_ingredients_envase_requires_weight ON public.master_ingredients IS
    '[P1-PLAN-LOTE-791] Una fila que se vende en envase (market_container o default_unit de envase) exige container_weight_g > 0. Sin él, la lista rotula el envase a peso («1 lb de Azafrán»). Si tu INSERT revienta aquí: rellena el envase, no relajes la constraint.';

DO $$
DECLARE _existe int;
BEGIN
    SELECT count(*) INTO _existe FROM pg_constraint
    WHERE conname = 'master_ingredients_envase_requires_weight'
      AND conrelid = 'public.master_ingredients'::regclass;
    IF _existe <> 1 THEN
        RAISE EXCEPTION '[P1-PLAN-LOTE-791] la constraint no quedó creada';
    END IF;
END $$;
