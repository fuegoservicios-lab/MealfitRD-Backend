# 3.0-F4 — `IngredientLine`: compras y macros, no el texto

`[P1-ARQ30-F4-CANONICAL · 2026-09-06]` · gap `ARQ30-P1-01`

## Qué se hizo, y qué NO

`ARQ30-P1-01` propone una autoridad única sobre la composición —`IngredientLine` / `RecipeVariant` /
`PlanRevision`— **de la que se deriven texto y compras**. El propio encargo describe ese movimiento
como `expand → verificación/canary → promoción → retirada`, y dice literalmente que no se borre la
vía anterior en el mismo paso que transfiere su autoridad.

Esto es el **`expand`**: [`canonical_recipe.py`](../canonical_recipe.py), una representación de
**solo lectura** que no cambia ni una línea de lo que se entrega hoy. Un test lo ancla —
`test_nadie_en_produccion_escribe_con_esta_representacion` falla en cuanto un módulo de producción la
importe.

## La medición decidió el alcance

Antes de mover nada se midió si la representación aguanta lo que el motor escribe, sobre **11.073
líneas de 96 planes vivos** (`scripts/canonical_roundtrip.py`):

| | |
|---|---|
| roundtrip **exacto** | **0,5 %** — 50 |
| roundtrip **equivalente** (normalizando caja y acentos) | **17,8 %** — 1.973 |

Y lo que se pierde no es formato:

```
3 huevos                    →  3 unidad de Huevo
45 g de melón en cubos      →  45 g de Melón          ← «en cubos» desaparece
3 dientes de ajo            →  3 diente de Ajo
1¼ cucharadas de cilantro   →  1.25 cda de Cilantro
```

El **corte** es información culinaria de la que dependen los pasos de la receta; el plural natural y
el tamaño («mediana») son lo que hace que una receta se lea como escrita por alguien.

> **Decisión del dueño, 2026-09-06: la representación sirve para COMPRAS y MACROS. El texto sigue
> siendo del LLM y de la tubería que ya lo escribe.**

Está escrito en la cabecera del módulo y anclado por un test, porque sin eso alguien «completaría» el
roundtrip dentro de seis meses creyéndolo una tarea pendiente — y degradaría todas las recetas.

## En ese alcance, sí aguanta

| | |
|---|---|
| resuelve al catálogo | **99,5 %** — los 60 que no son `agua`, `hielo`, `cubos_de_hielo`: no son alimentos |
| gramos derivables | **82,4 %** |
| cantidad conservada | **96,3 %** |

**Los gramos subieron del 19,9 % al 82,4 % sin escribir una regla nueva.** La primera versión paraba
en la conversión de unidades; una taza de espinacas, tres huevos o dos rebanadas de pan necesitan la
**densidad del catálogo**, y esa conversión ya tenía dueño: `nutrition_db.to_grams`. Que además lleva
dentro la lección de `P1-UNKNOWN-UNIT-NOT-WHOLE` — una unidad desconocida **no** es una unidad entera
del alimento; para una hierba eso es el mazo, y de ahí salían los 415 g de cebollín.

## Lo que el módulo NO hace, a propósito

**No parsea de cero.** Compone `_parse_quantity`, `_reconcile_qty_with_gram_hint`,
`canonical_units.canonicalize_unit`, `nutrition_db.to_grams` e `ingredient_id_for`. Un quinto parser
sería la deriva que este repo ya pagó con `canonicalize_diet_type` y con `pantry_names_match`; hay un
test por cada autoridad compuesta.

**No decide gramos**: guarda lo que la autoridad respondió **y quién respondió** (`grams_source`).
Cuando `ARQ30-P1-04` unifique la autoridad de cantidades, cambia el proveedor y no el contrato.

**No inventa el estado.** `desconocido` **no es** `crudo`: el rendimiento cocido↔seco de legumbres es
0,35×, así que asumir crudo donde el texto calla triplicaría la compra. Misma invariante que I20.

## Dos defectos que la medición destapó

1. **La pista de gramos vivía dentro del nombre.** «1 batata mediana cocida, 250 g» producía el id
   `batata_250_g`: dos líneas del mismo alimento con pistas distintas daban identidades distintas, y
   la lista de compras las habría contado aparte.
2. **El render duplicaba el estado.** `_parse_quantity` deja el calificativo en el nombre, así que
   anexarlo producía «pasta integral seca **seco**» — inventaba una palabra que el original no traía.

Anotado y no tocado: `Ñame` produce el `ingredient_id` `name`, porque `strip_accents` convierte la ñ
en n. Colisión latente con cualquier alimento inglés llamado «name»; hoy no existe ninguno.

## Siguiente paso

`ARQ30-P1-04` (una autoridad de cantidades) es el consumidor natural: `grams_source` existe para que
ese cambio se vea desde fuera. La promoción a compras reales necesita antes un canary por cohorte,
como el gap exige.

Test: [`test_p1_arq30_f4_canonical.py`](../tests/test_p1_arq30_f4_canonical.py).
Sonda: [`scripts/canonical_roundtrip.py`](../scripts/canonical_roundtrip.py).

---

# 3.0-F5 — Factibilidad conjunta: decisión medida de NO adoptar

`[P1-ARQ30-F5-NO-ADOPT · 2026-09-06]` · gap `ARQ30-P1-02`

## La limitación es real y está reproducida

`_feasibility_report` calcula una cota **por coordenada**. El contraejemplo que la auditoría propone,
ejecutado tal cual:

```
entries = [{macros: {protein: 20, fats: 10}}]     x ∈ [0,5 · 2]
target  = {protein: 40, fats: 5}
_feasibility_report(...) → {}                      ← «todo alcanzable»
```

Proteína 40 exige x=2; con x=2 la grasa es 20, no 5. Cada coordenada cabe sola, **conjuntamente no**.
La cota es necesaria, no suficiente. Está anclado en `test_el_contraejemplo_sintetico_de_la_auditoria`
como el gap exige.

## Y aun así no se adopta

El gap pone las dos condiciones él mismo: «primero cerrar `ARQ27-P1-08` y medir el residual» y
«adoptar factibilidad conjunta **solo con mejora demostrada**». `ARQ27-P1-08` está cerrado, así que se
midió (30 días, `scripts/solver_residual_probe.py`):

| | |
|---|---|
| comidas dimensionadas | 3.217 |
| no convergieron | 1.224 — **38,0 %** |
| declaradas infactibles | 326 — **10,1 %** |
| corridas con ≥1 infactible | 156 de 264 — 59,1 % |

Cifras altas. Y entonces la pregunta que decide —el criterio «no gastar LLM regenerando con el mismo
conjunto inviable»—, emparejando por `session_id`:

| | con infactibles (6.706) | sin (4.018) |
|---|---|---|
| `attempts` medio | **1,552** | **1,544** |
| `review_passed` | 92,4 % | 93,0 % |
| desviación calórica p50 / p90 | 0,020 / 0,036 | 0,019 / 0,035 |
| fallback | 2,5 % | 3,7 % |

**La infactibilidad no cuesta ni un reintento ni un punto de calidad**, y las corridas que la sufren
caen *menos* al fallback. El cerrador y los clamps la absorben. El daño que el criterio persigue no
está ocurriendo, así que no hay mejora que obtener de un LP/MILP/CP-SAT.

> «Si el prototipo no mejora, conserva el actual y registra el resultado experimental; no cambies de
> librería solo para usar el nombre 3.0.» — el propio encargo.

`test_no_se_anadio_ninguna_libreria_de_optimizacion` ancla que no entró ninguna.

## El criterio que sí faltaba: a qué NIVEL vale el testigo

«Un testigo local no certifica todos los niveles.» Medido: `_feasibility_report` se invoca **solo
desde `portion_solver`, siempre a nivel de COMIDA**. En `horizon.py` y `plan_policy.py` no hay
factibilidad. Un `_solver_infeasible` vacío dice que ESA comida es dimensionable — **no** que el día
ni el horizonte lo sean.

⚠️ Existe además `_pantry_feasibility_report`, que **es otra cosa**: la factibilidad de la NEVERA
(«¿cuántos días aguanta la despensa?»), esa sí a nivel de día. Confundirlas sería leer una garantía de
la despensa como una garantía del solver. Un `grep` sin frontera de palabra las mezcla — pasó al
escribir este test.

## Cuándo reabrir

Corre `python scripts/solver_residual_probe.py`. Si las dos columnas se separan —`attempts` o
desviación—, la decisión cambia. Es de los números, no de la opinión.

Test: [`test_p1_arq30_f5_no_adopt.py`](../tests/test_p1_arq30_f5_no_adopt.py).
