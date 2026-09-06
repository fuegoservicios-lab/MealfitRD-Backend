# Roadmap 2.7 · Fase F2 — la capacidad vegetal que ya existía y nadie alcanzaba

**[P1-ARQ27-F2 · 2026-09-06]** ARQ27-P1-03. El catálogo tenía las proteínas vegetales desde hacía
meses; lo que faltaba eran las recetas que las nombraran.

## El hallazgo

| Alimento | Fila del catálogo público | Usos como constituyente (antes) | Ahora |
|---|---|---:|---:|
| Tofu firme | sí, categoría Proteínas | **0** | 10 |
| Soya texturizada | sí, 51,5 g de proteína / 100 g | **0** | 8 |
| Leche de soya | sí, categoría Lácteos | **0** | 16 |
| Edamame | sí, categoría Proteínas | 1 (solo US) | 8 |

Con tofu en cero, la familia `Tofu` del pool vegano —un quinto del pool— programaba días que el
registro no podía servir. La directiva histórica «sin tofu: no se vende» convivía con una fila
`Tofu firme` en el catálogo y una familia `Tofu` en el motor.

## Lo que cambió, medido con el embudo

Candidatos **mínimos por franja** (`scripts/coverage_funnel.py`, guarda real sobre constituyentes):

| Biblioteca | vegano antes | vegano ahora | vegano sin gluten antes | ahora |
|---|---:|---:|---:|---:|
| DO | 4 | **7** | 2 | 5 |
| PR | 4 | **7** | 1 | **7** |
| MX | 3 | **7** | 2 | 6 |
| CO | 3 | **7** | 4 | **7** |
| ES | **1** | **7** | 1 | 6 |
| US | 3 | **7** | 1 | 6 |

El piso de 7 sale de la cuenta del anexo de catálogo: 7 ocasiones de una franja sin repetir exigen 7
recetas distintas. Vegetariano subió de 7-10 a 10-13 de rebote, porque todo lo vegano lo es también.

88 platos nuevos en dos tandas: 60 de capacidad general y 28 desayunos sin gluten.

## Las dos cosas que solo aparecieron al medir

### `Avena` lleva la clase `gluten`

La primera tanda dejó `vegano` en ≥7 y destapó el cuello real: los desayunos veganos se apoyaban en
pan y avena, y **un vegano celíaco tenía UN desayuno en PR, ES y US**. La segunda tanda se construyó
sobre quinoa, maíz, víveres y fruta — y ninguna de las dos habría existido sin medir después de la
primera. Comprobado sobre el catálogo: `Avena` → `['gluten']`, `Quinoa` → `[]`, `Granola` → `[]`,
`Harina de maíz precocida` → `[]`.

### Dos bibliotecas beta usaban el embutido dominicano

`Frijoles charros con longaniza` (MX) y `Longaniza santarrosana con papa` (CO) llevaban **`Longaniza
dominicana`** teniendo `Chorizo mexicano` y `Chorizo santarrosano` en el catálogo. El nombre del plato
ya lo decía —«charros», «santarrosana»—: el ingrediente se copió del molde dominicano y nadie lo
revisó, porque hasta ARQ27-P1-07 nada comprobaba el mercado. Corregidos ingrediente **y** nombre: un
título que promete longaniza y sirve chorizo es la misma clase de defecto que `P1-NAME-SPECIFICITY`
persigue en los planes vivos.

## Tres decisiones del diseño

1. **Las familias nuevas se llaman como la FILA del catálogo.** `_FAMILIES_BY_DIET["vegan"]` gana
   `"Soya texturizada"` y `"Edamame"`, no `"Soya"`. Se resuelven por el puente de etiqueta genérica
   (`legumbre`, ARQ27-P1-01) mirando los constituyentes, sin inventar tokens de familia. Un `"Soya"` a
   secas habría alcanzado **«Salsa de soya»** —un condimento de 8 g de proteína— y lo habría puesto de
   proteína protagonista del día.
2. **Ninguna ola queda cubierta exclusivamente con soja.** Quitando la soja siguen quedando ≥5
   desayunos veganos en las seis cocinas; hay un test que lo exige. Una alergia a la soja elimina una
   familia entera, así que la variedad que solo existe con soja es variedad aparente.
3. **El preflight bloquea el alta, no la corrige.** Antes de escribir nada se comprueba que cada
   ingrediente resuelve contra el catálogo vivo, no viola la dieta vegana, se vende en el mercado de
   su biblioteca y trae gramos > 0. Una plantilla con un ingrediente sin resolver nacería `partial`
   tras ARQ27-P0-02 y el selector no la ofrecería jamás: el alta sería trabajo perdido y silencioso.

## Reglas del repo que las altas tuvieron que respetar

- **Arroz/pasta como BASE nunca en desayuno ni cena** (SSOT con la rúbrica dura del juez culinario).
  Cinco cenas nuevas la rompían. Se cambió el acompañamiento por un tubérculo —batata, yuca, papa— en
  vez de recolocar la etiqueta: mover el rótulo habría mentido sobre lo que el plato es, y cinco cenas
  más de arroz tampoco eran variedad.
- El baseline del corpus DO (`scripts/data/do_corpus_retarget_baseline_2026_08_18.json`) se regeneró:
  1.521 → 1.533 entradas, **solo altas**, `accepted_deltas` intacto en 4. Ningún mapeo previo cambió.

## Ficheros y tests

`data/dish_templates*.json` (los 6) · `data/registry/*` recompilados ·
[`horizon.py`](../horizon.py) `_FAMILIES_BY_DIET["vegan"]`.

Test: [`test_p1_arq27_f2_veganos.py`](../tests/test_p1_arq27_f2_veganos.py) — 26 casos que miran el
**selector y el pool**, no el fichero de plantillas. Es deliberado: entre `7b6df93` y `8d83abb` se
añadieron 23 plantillas y los IDs que el blueprint ofrecía fueron exactamente los mismos antes y
después. Añadir platos no es lo mismo que llegar a la decisión.

```bash
python -m pytest backend/tests/test_p1_arq27_f2_veganos.py -q
python backend/scripts/coverage_funnel.py
```

---

# ARQ27-P1-09 — conservación según estado, envase y equipo

Mismo P-fix (`P1-ARQ27-F2`), otro gap. `pantry_durability` decidía por el ALIMENTO y su categoría, y
le faltaban dos dimensiones que sí cambian la logística de una compra única.

## 1. Un congelado de fábrica exige congelador

Cuatro filas —**Edamame, Papas ralladas, Wafles, Bolitas de papa**— estaban clasificadas `pantry` 90
con un comentario al lado que decía «congelados de fábrica». La tabla afirmaba que un paquete de
edamame aguanta tres meses en la alacena, así que en un ciclo de una sola compra **sin congelador**
pasaban el guard el día 30 sin que nada avisara.

Clase nueva `frozen`: **1 día fuera, 365 dentro**. Y código de issue propio, `frozen_needs_freezer`,
porque el consejo no es el mismo: a la proteína fresca se le ofrece una alternativa de despensa; a un
congelado de fábrica hay que decirle que sin congelador ese plato no cabe en su compra.

| Alimento | antes | ahora |
|---|---|---|
| Edamame | `pantry` 90 / 90 | `frozen` 1 / 365 |
| Papas ralladas, Wafles, Bolitas de papa | `pantry` 90 / 90 | `frozen` 1 / 365 |

Ciclo de 30 días, día 25, sin congelador: `frozen_needs_freezer`. Con congelador: pasa. Un guard que
bloquea siempre no informa — bloquearía justo a quien puede permitírselo.

## 2. Una bebida estable cerrada no lo es abierta

La leche vegetal era 365 días, abierta o cerrada. Mecanismo idéntico al de `fresh_state`
(P1-DURABILITY-FRESH-STATE): **el calificativo del nombre manda sobre la tabla**. `Leche de soya` →
despensa 365; `Leche de soya abierta` → frío 7, regla `opened_package`. 16 alimentos en la lista
(leches vegetales y evaporada, conservas abiertas, aceitunas, salsa de tomate, atún en agua).

## Las dos fronteras que se declaran en vez de fingirse

- **No se deduce solo que un cartón abierto el día 1 ya no sirve el día 20** de la misma compra. Eso
  exige saber en qué días se usa cada ingrediente, y pertenece al modo «cocino por tandas», que este
  módulo aún no representa.
- **`Lentejas cocidas` sigue siendo despensa 180 a propósito.** El módulo responde «¿cuánto aguanta lo
  que el usuario COMPRA?», y lo que compra es lenteja seca; el plato se cocina ese día. Tratar cada
  nombre cocinado como sobras de nevera bloquearía platos correctos. Esa frontera ya estaba declarada
  antes de este gap —en la nota de «arroz cocido»— y se respeta.

El criterio de cierre del gap pedía además distinguir «lentejas secas, cocidas, en conserva cerrada y
abierta». Las dos últimas quedan cubiertas (`opened_package`); las dos primeras son la frontera de
arriba, y cerrarlas de verdad es trabajo del modo por tandas, no de esta tabla.

Test: [`test_p1_arq27_f2_conservacion.py`](../tests/test_p1_arq27_f2_conservacion.py) — 24 casos,
incluidos los que anclan que las cuatro clases anteriores y la ventana de congelación no se movieron.

---

# ARQ27-P1-08 — la señal del solver llega a las reparaciones

`_solver_infeasible` existe desde `P3-SOLVER-FEASIBILITY` y dice, por macro y con dirección, que **no
hay solución con estos alimentos dentro del clamp**: `{'protein': 'high'}` = falta un portador de
proteína y ninguna re-escala lo arregla.

Y aun así `_close_protein_gap_for_meal` empezaba SIEMPRE por `_try_scale_existing_protein`, o sea por
la operación que el solver acababa de demostrar imposible. Medido el 06-sep sobre 1.174 platos de 200
planes vivos: 86 infactibles, **34 con `protein:high`**. Dos capas trabajando sobre la misma señal sin
compartirla — la recomendación explícita del propio análisis del solver, por encima de invertir una
semana en factibilidad conjunta.

**Saltarse el escalado no abre ningún hueco.** `_try_scale_existing_protein` es «cierre completo o
nada» desde `P3-PROTEIN-CLOSER-SCALE-FIRST` (antes un crecimiento parcial marcaba `_protein_closed` y
dejaba el déficit abierto en perfiles bariátricos). Lo que se ahorra es la pasada inútil; lo que se
gana es que el motivo quede en el log y en `_closer_used_solver_signal`, para poder repetir la
medición.

## Dos errores que el test vigila

- **Reaccionar a `_solver_not_converged`** en vez de a `_solver_infeasible` saltaría el escalado en el
  **41 %** de los platos, y casi todos podían crecer perfectamente. No converger es la norma —el clamp
  por línea rara vez clava el target y los closers terminan el trabajo—; no haber solución es el 7 %.
- **Leer la señal después del escalado**, que la dejaría inerte.

## Lo que sigue sin consumidor, dicho en vez de fingido

`fats:low` (20 platos), `carbs:high` (16), `fats:high` (16) y `carbs:low` (7) no tienen a quién
dárselos: no existe un cerrador de grasa ni de carbohidrato. Fabricar uno sin medir antes si mejora el
plato es justo lo que el análisis del solver desaconseja. Queda declarado en
[`solver_estado_terminacion.md`](solver_estado_terminacion.md), no cerrado en falso.

Test: [`test_p1_arq27_f2_solver_signal.py`](../tests/test_p1_arq27_f2_solver_signal.py) — 11 casos.

---

# ARQ27-P1-05 — identidad ≠ categoría comercial

Cinco filas de nombre vegetal viven en la categoría **Lácteos** del catálogo: Leche de soya, de coco,
de avena, de almendras y Yogur de coco. La categoría es de TIENDA —dice en qué pasillo está—, y el
gap pregunta si el motor la lee como verdad dietaria.

**Medido antes de tocar nada, sobre las 347 filas del catálogo vivo.** Casi todo estaba ya bien: el
guard de dieta acertaba en las nueve pruebas, la durabilidad resolvía por nombre y no por categoría
(las leches vegetales son despensa 365, no el `fresh 10` del default de «lacteos»), ninguna plantilla
vegana ofrecía un lácteo real, y los alias no inflaban el recuento de identidades (347 filas, ~1.300
alias, 347 identidades).

Quedaba **una** fila. `Yogur de coco` se declaraba `['lacteos', 'lactosa']`: la lista de excepciones
de `P1-PLANT-MILK-NOT-DAIRY` traía «yogur vegetal» y no el nombre real de la fila del catálogo. Un
plato con yogur de coco caía fuera del pool de un alérgico a la leche por un alérgeno que no tiene.

## El defecto de fondo no era la fila: eran las dos capas

`allergen_classes_for` (lo que el registry **declara**, y que alimenta `intrinsic_risk_attributes` y
con él el `exclude_allergens` del selector) y `_scan_allergen_violations` (lo que el guard **decide**)
son dos caminos hacia la misma pregunta, y divergieron en una fila sin que nadie se enterara: cada
capa por separado parecía razonable.

`test_las_dos_capas_de_lacteo_coinciden` recorre el catálogo entero y las obliga a estar de acuerdo.
Esa es la parte que impide la próxima; la fila era solo el síntoma.

Test: [`test_p1_arq27_f2_identidad.py`](../tests/test_p1_arq27_f2_identidad.py) — 35 casos, incluidos
los que anclan que la excepción no se abrió de más (siete lácteos reales siguen siéndolo) y que una
bebida vegetal conserva SUS alérgenos (la de almendras sigue siendo frutos secos; la de avena,
gluten).
