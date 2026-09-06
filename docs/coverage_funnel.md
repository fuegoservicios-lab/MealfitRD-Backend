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

## Medición del 2026-09-05

```
EMBUDO DE COBERTURA — cuántos platos sobreviven a los filtros de un usuario

── dominican_criolla  (mercado DO) ─────────────────────────────
   escenario                       desayuno  almuerzo  merienda      cena    mínimo
   base                                  29        51        34        39        29
   vegetariano                           27         8        32        15         8
   vegano                                 9         6        20         2         2  ⚠
   veg + sin lácteos                      8         8        11         7         7
   veg + sin lácteos ni huevo             0         4         6         1         0  ⚠
   sin mariscos ni pescado               28        39        34        28        28
   sin gluten                            15        42        21        37        15
   día 25 sin congelador                  9        10         8         8         8
   veg · día 25 sin congelador            9         5         8         5         5
   día 25 congelador limitado             9        10         8         8         8

── puertorico_criolla  (mercado PR) ────────────────────────────
   escenario                       desayuno  almuerzo  merienda      cena    mínimo
   base                                  26        42        23        30        23
   vegetariano                           24         9        18         8         8
   vegano                                 8         6        13         0         0  ⚠
   veg + sin lácteos                      8         9         8         2         2  ⚠
   veg + sin lácteos ni huevo             0         5         8         0         0  ⚠
   sin mariscos ni pescado               25        30        18        19        18
   sin gluten                            11        40        17        27        11
   día 25 sin congelador                  8        10         7         9         7
   veg · día 25 sin congelador            8         6         6         4         4
   día 25 congelador limitado             8        10         7         9         7

── mexico_casera  (mercado MX) ─────────────────────────────────
   escenario                       desayuno  almuerzo  merienda      cena    mínimo
   base                                  28        43        25        35        25
   vegetariano                           26        11        25        17        11
   vegano                                10         8        19         5         5
   veg + sin lácteos                      9         8         9         7         7
   veg + sin lácteos ni huevo             0         5         9         3         0  ⚠
   sin mariscos ni pescado               28        34        25        27        25
   sin gluten                            18        42        18        32        18
   día 25 sin congelador                  7         9         6         8         6
   veg · día 25 sin congelador            7         5         6         6         5
   día 25 congelador limitado             7         9         6         8         6

── colombia_casera  (mercado CO) ───────────────────────────────
   escenario                       desayuno  almuerzo  merienda      cena    mínimo
   base                                  27        45        27        36        27
   vegetariano                           22        11        26        17        11
   vegano                                 5        10        18         5         5
   veg + sin lácteos                      6        11        10         8         6
   veg + sin lácteos ni huevo             0         6        10         5         0  ⚠
   sin mariscos ni pescado               26        32        27        27        26
   sin gluten                            20        43        25        35        20
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
   veg + sin lácteos                      6         6         4         3         3
   veg + sin lácteos ni huevo             2         3         3         3         2  ⚠
   sin mariscos ni pescado               26        31        17        26        17
   sin gluten                             8        23        11        28         8
   día 25 sin congelador                  8        10         7         8         7
   veg · día 25 sin congelador            8         7         6         7         6
   día 25 congelador limitado             8        10         7         8         7

⚠ = alguna franja baja de 3 candidatos · × = alguna franja se queda sin ninguno
«elegibles» = pasa franja, alergias, dieta y conservación. NO incluye solver, precio ni cuotas.
```

## Lo que dice esta primera medición

Tres huecos reales, y ninguno es «faltan platos en general»:

1. **Vegano en Puerto Rico: cero cenas.** No hay ninguna. Es el único cero que produce la dieta sola, sin
   alergias encima.
2. **Vegetariano sin lácteos ni huevo: cero desayunos** en DO, PR, MX y CO — cuatro de las seis bibliotecas. El
   desayuno criollo se sostiene sobre huevo y queso, así que quitarle los dos lo vacía.
3. **El almuerzo vegetariano es estrecho en las cocinas criollas**: 8 en DO, 8 en PR, 11 en MX y CO, contra 29
   en España. No es un cero, pero con las cuotas del horizonte encima deja poco margen de variedad.

Lo que **no** aparece: ningún hueco por conservación. En el día 25 sin congelador quedan 6-8 candidatos por
franja en las seis bibliotecas — poco, pero no cero.

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
