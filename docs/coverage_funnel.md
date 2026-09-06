# Embudo de cobertura del catálogo de platos

**[P1-COVERAGE-FUNNEL · 2026-09-05]** Cuántas plantillas sobreviven de verdad a los filtros de un usuario.

## Por qué existe

Las seis bibliotecas suman **667 entradas** —123 DO, 118 CO, 110 PR, 109 MX, 104 ES, 103 US— y ese número no
dice nada útil por sí solo: nadie había medido cuántas quedan después de la franja, las alergias, la dieta y la
conservación. Sin esa medición, «faltan platos» y «se eligen mal» son indistinguibles, y escribir más plantillas
es apostar.

## Cómo se vuelve a medir

```bash
python scripts/coverage_funnel.py                        # la tabla
python scripts/coverage_funnel.py --json                 # para diffear entre versiones
python scripts/coverage_funnel.py --desglose vegetariano # etapa a etapa
```

«Elegibles» = pasa franja, alergias, dieta y conservación. **No** incluye si el solver puede dimensionar el
plato con esos gramos, ni el precio, ni las cuotas del horizonte: un candidato que sobrevive aquí es elegible,
no necesariamente servible.

## Medición del 2026-09-05 (antes de cerrar los huecos)

Tres huecos reales, y ninguno era «faltan platos en general»:

1. **Vegano en Puerto Rico: cero cenas.** El único cero que producía la dieta sola.
2. **Vegetariano sin lácteos ni huevo: cero desayunos** en DO, PR, MX y CO — cuatro de las seis bibliotecas. El
   desayuno criollo se sostiene sobre huevo y queso, así que quitarle los dos lo vaciaba.
3. **Almuerzo vegetariano estrecho en las cocinas criollas**: 8 en DO y PR, 11 en MX y CO, contra 29 en España.

Lo que **no** aparecía: ningún hueco por conservación. En el día 25 sin congelador quedaban 6-8 candidatos por
franja en las seis bibliotecas — poco, pero no cero.

## Medición del 2026-09-06 (después de P1-GAP-DISHES-VEG)

23 plantillas nuevas, dirigidas a esos dos huecos, y una corrección de etiquetado que resultó valer tanto como
los platos: **«Leche de coco» y «Mantequilla de maní» se declaraban LÁCTEOS**, así que desaparecían del plan de
cualquier alérgico a la leche y mentían a quien sí los tolera.

```
EMBUDO DE COBERTURA — cuántos platos sobreviven a los filtros de un usuario

── dominican_criolla  (mercado DO) ─────────────────────────────
   escenario                       desayuno  almuerzo  merienda      cena    mínimo
   base                                  32        52        34        43        32
   vegetariano                           30         9        32        19         9
   vegano                                12         7        20         6         6
   veg + sin lácteos                     12         9        13        11         9
   veg + sin lácteos ni huevo             4         5         8         5         4
   sin mariscos ni pescado               31        40        34        32        31
   sin gluten                            17        43        21        41        17
   día 25 sin congelador                 10        11         8         8         8
   veg · día 25 sin congelador           10         6         8         5         5
   día 25 congelador limitado            10        11         8         8         8

── puertorico_criolla  (mercado PR) ────────────────────────────
   escenario                       desayuno  almuerzo  merienda      cena    mínimo
   base                                  29        43        23        35        23
   vegetariano                           27        10        18        13        10
   vegano                                11         7        13         5         5
   veg + sin lácteos                     13        10        10         7         7
   veg + sin lácteos ni huevo             5         6        10         5         5
   sin mariscos ni pescado               28        31        18        24        18
   sin gluten                            12        41        17        32        12
   día 25 sin congelador                  8        10         7        10         7
   veg · día 25 sin congelador            8         6         6         5         5
   día 25 congelador limitado             8        10         7        10         7

── mexico_casera  (mercado MX) ─────────────────────────────────
   escenario                       desayuno  almuerzo  merienda      cena    mínimo
   base                                  31        43        25        35        25
   vegetariano                           29        11        25        17        11
   vegano                                13         8        19         5         5
   veg + sin lácteos                     12         8         9         7         7
   veg + sin lácteos ni huevo             3         5         9         3         3
   sin mariscos ni pescado               31        34        25        27        25
   sin gluten                            20        42        18        32        18
   día 25 sin congelador                  7         9         6         8         6
   veg · día 25 sin congelador            7         5         6         6         5
   día 25 congelador limitado             7         9         6         8         6

── colombia_casera  (mercado CO) ───────────────────────────────
   escenario                       desayuno  almuerzo  merienda      cena    mínimo
   base                                  30        45        27        36        27
   vegetariano                           25        11        26        17        11
   vegano                                 8        10        18         5         5
   veg + sin lácteos                      9        11        11         8         8
   veg + sin lácteos ni huevo             3         6        11         5         3
   sin mariscos ni pescado               29        32        27        27        27
   sin gluten                            22        43        25        35        22
   día 25 sin congelador                  7        11         7         8         7
   veg · día 25 sin congelador            6         5         7         6         5
   día 25 congelador limitado             7        11         7         8         7

── spain_mediterranea  (mercado ES) ────────────────────────────
   escenario                       desayuno  almuerzo  merienda      cena    mínimo
   base                                  27        52        23        40        23
   vegetariano                           26        29        21        27        21
   vegano                                12        26        13        11        11
   veg + sin lácteos                     14        29        12        22        12
   veg + sin lácteos ni huevo             4        25         9         8         4
   sin mariscos ni pescado               26        31        21        26        21
   sin gluten                            10        47        14        36        10
   día 25 sin congelador                  8        14         6        11         6
   veg · día 25 sin congelador            8         7         5         8         5
   día 25 congelador limitado             8        14         6        11         6

── us_everyday  (mercado US) ───────────────────────────────────
   escenario                       desayuno  almuerzo  merienda      cena    mínimo
   base                                  26        41        18        33        18
   vegetariano                           26         8        16        12         8
   vegano                                12         5        10         5         5
   veg + sin lácteos                      8         7         7         3         3
   veg + sin lácteos ni huevo             4         4         6         3         3
   sin mariscos ni pescado               26        31        17        26        17
   sin gluten                             8        23        11        28         8
   día 25 sin congelador                  8        10         7         8         7
   veg · día 25 sin congelador            8         7         6         7         6
   día 25 congelador limitado             8        10         7         8         7

⚠ = alguna franja baja de 3 candidatos · × = alguna franja se queda sin ninguno
«elegibles» = pasa franja, alergias, dieta y conservación. NO incluye solver, precio ni cuotas.
```

**Ninguna franja baja de 3 candidatos en ningún escenario de ninguna biblioteca.** Los ceros desaparecieron: el
vegano puertorriqueño tiene 5 cenas y el vegetariano sin lácteos ni huevo tiene desayuno en las seis cocinas.

Los platos nuevos respetan la regla editorial de la biblioteca —una cena no se declara con base de almidón—: el
plato cuya identidad ES el arroz (moro, arroz con gandules) se declaró almuerzo, que es su sitio, y el guiso que
se sirve con arroz declara base «legumbre», que es lo que estructura el plato.

## Dos trampas que ya costaron una medición falsa

- El perfil cultural guarda su mercado en **`market_default`**, no en `country`. Con `country` la función
  devolvía `None`, las seis bibliotecas cargaban el snapshot por defecto y **la tabla salía seis veces idéntica**
  sin que nada avisara.
- Los alérgenos del registry están **en español y con dos formas cada uno** (`lacteos`/`lactosa`,
  `huevo`/`huevos`). Filtrar por `dairy` deja el filtro inerte y las columnas salen iguales a las de «sin
  alergias» — igual de creíbles y completamente falsas.

Las dos las caza `tests/test_p1_coverage_funnel.py`: lo que se vigila es **el instrumento**, no las cifras. Si
mañana hay más platos, los números cambian y el test sigue verde; si el instrumento vuelve a medir de mentira,
cae.
