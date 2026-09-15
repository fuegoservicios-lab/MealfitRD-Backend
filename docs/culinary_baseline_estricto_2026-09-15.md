# Línea base ESTRICTA del instrumento culinario · 2026-09-15

`P1-PLAN-LOTE-60` (lote 38 del [plan 38-44](plan_agente_lotes_38_43_2026_09_14.md) · C1 cierre). **Esta cifra no se toca
después**: es contra la que se miden los lotes 39 (detectores) y 40 (calibración del juez). Si el instrumento cambia, se
publica otra línea base con su fecha, al lado de ésta.

## Qué se midió

- **Verdad humana**: la anotación con rúbrica del dueño, [`culinary_golden_anotaciones_angelo.json`](culinary_golden_anotaciones_angelo.json)
  (80/80, ciega: 66 defecto · 5 dudoso · 9 ok; 75 casos puntuables). Un solo anotador: sin kappa y sin promoción, que
  exige dos. Ningún «defecto» queda sin rúbrica (el único, `060b4fda6a`, se transcribió de la nota del dueño del 09-07).
- **Máquina, antes**: las columnas `maquina_determinista` / `maquina_juez` del golden set, que son los hallazgos
  PERSISTIDOS en 96 planes el 2026-09-06 (sólo V1-V5; el juez anterior a `[dudosa]`).
- **Máquina, después**: `maquina_determinista_2026-09-15` / `maquina_juez_2026-09-15`, escritas por
  [`scripts/culinary_golden_refresh.py`](../scripts/culinary_golden_refresh.py) con el escáner y el juez de hoy (sobre el commit
  `14f8d8f7` más los cambios de este lote aún sin commitear, que no tocan el escáner —por eso el refresco anota `14f8d8f7+cambios_sin_commit`—; reglas `c3a77b84d992801b`, catálogo `master_ingredients` de 349 filas con
  huella `1f4f95b33a191dfe` — la misma que el corpus fijo del 09-12, así que `--catalogo scripts/data/culinary_corpus_2026_09_12.json`
  reproduce la columna sin base). Cada hallazgo lleva al final el alimento que acusa —`(alimento: X)`, el `food` de
  la violación, que el builder del 09-06 tiraba; en el juez, `(componente: X)`—.
- **El mismo adjudicador en las dos**, el de este lote: el alimento del dueño sólo restringe si el hallazgo nombra uno. El
  JSON de al lado es la salida literal de

      python scripts/culinary_golden_score.py --estricto --anotaciones docs/culinary_golden_anotaciones_angelo.json \
          --comparar-maquina 2026-09-15 --json

## Por capa

| | capa | tp/fp/fn | precisión % | recall % | precisión pond. | recall pond. | IC95 precisión | IC95 recall | por código / código+alimento |
|---|---|---|---|---|---|---|---|---|---|
| antes (09-06) | determinista | 11/30/36 | 26,8 | 23,4 | 32,4 | 8 | 10,7–40,7 | 9,4–32,7 | 8 / 3 |
| antes (09-06) | juez | 2/32/7 | 5,9 | 22,2 | 5,7 | 7,9 | 0–13,3 | 0–50 | 0 / 2 |
| después (2026-09-15) | determinista | 18/60/29 | 23,1 | 38,3 | 28,6 | 40,6 | 12,7–31,5 | 20–51 | 0 / 18 |
| después (2026-09-15) | juez | 0/28/9 | 0 | 0 | 0 | 0 | 0–0 | 0–0 | 0 / 0 |

tp/fp/fn en crudo; «pond.» corrige el muestreo estratificado (25 de 919 comidas «sin hallazgo»); IC95 por conglomerado
(el plan), 1.000 remuestreos con semilla fija. La última columna dice cómo se emparejó cada acierto. Las quejas `[dudosa]`
del juez se observan y no puntúan (7 en la columna refrescada).

Con el adjudicador anterior a este lote la misma anotación daba **0/41/47** (determinista) y **0/34/9** (juez): exigía el
`alimento` del dueño como subcadena del texto de la máquina, y el de V4 no nombra ninguno.

## Por clase del dueño

| clase del dueño | antes tp/fn | después tp/fn |
|---|---|---|
| `cantidad_inconsistente` | 7/12 | 10/9 |
| `estructura_del_plato` | 0/1 | 0/1 |
| `ingrediente_huerfano` | 2/3 | 3/2 |
| `masa_sobrante` | 0/1 | 0/1 |
| `nombre_no_corresponde` | 0/2 | 0/2 |
| `paso_incoherente` | 2/5 | 0/7 |
| `seco_sin_coccion` | 0/11 | 3/8 |
| `usa_lo_que_no_esta` | 1/8 | 1/8 |
| `verbo_alimento` | 1/0 | 1/0 |

Sin ningún código que las mecanice (FN del sistema entero, en las dos): `coccion_faltante` 8 y `otro` 2.

## Lo que dice de la máquina

- **El determinista de hoy ve más de lo que el dueño ve**: recall 23,4 → 38,3 %. Los
  aciertos nuevos son de los detectores que no existían el 09-06 —V7c caza 3 de los 11 «seco sin cocción» (0 antes), V6
  y V7e cantidades que V4 no leía—. Aciertos por código después: V7e 4, V3 3, V4 3, V6 3, V7c 3, V1 1, V5 1.
- **Y acusa más cosas que el dueño no marcó**: FP 30 → 60 (precisión 26,8 →
  23,1 %). Por código: V3 14, V7a 12, V1 9, V7e 7, V7b 5, V2 4, V6 4, V4 1, V5 1, V7c 1, V8a 1, V9 1. V3 (el ingrediente listado que
  ningún paso nombra) es la mayor fuente y ya lo era el 09-06; V7a entra con 12 y ninguno puntúa: la clase
  `lista_de_mas` existe en la rúbrica y el dueño no la usó (ver abajo).
- **El juez**: no acierta ninguno de los 9 defectos de su competencia que marcó el dueño (7 `paso_incoherente`, 2 `nombre_no_corresponde`): en 7 de esas 9 comidas no se quejó de nada; en `0eeebbee4d` vio el mismo problema que el dueño (la pechuga tratada como huevo, «bate pechuga de pollo con los 2 pechuga de pollo») pero lo llamó `tecnica_impropia`, otra frontera de la rúbrica; y en `2538fbe6a2` lo marcó `[dudosa]`, que se observa y no puntúa. Sus 28 FP son quejas donde el dueño no vio un defecto de esa clase (paso_incoherente 16, tecnica_impropia 6, nombre_no_corresponde 5, combo_absurdo 1). Antes daba 2/32/7 juzgando el plan entero; aquí juzga una comida por llamada, con menos contexto, y no es estable (16 de 69 comidas cambian de quejas entre dos corridas). Es el punto de partida de la calibración del lote 40, no un veredicto sobre el juez de producción.

## Cómo se midió el juez y cuánto costó

- Modelo de producción (glm-5.3-flash; en `llm_usage_events`, 67 llamadas del juez en 14 días y todas de ese modelo),
  rúbrica DO, sin formulario, **una comida por llamada**: el golden set no guarda el resto del día y en producción el
  juez ve el plan entero. No es exactamente el juicio de producción, y la diferencia se declara.
- `MEALFIT_CULINARY_JUDGE_GUARD=warn` sólo en el proceso del script (no es el flip de C6, que es del dueño). Escrituras a
  la base sustituidas por dobles que cuentan: ninguna intentada.
- Corrida 1 (4 a la vez, timeout de producción 45 s): 71 comidas juzgadas, 9
  caídas por timeout, $0,0208 contados. Se repitió porque el texto de sus quejas no llevaba el
  componente. Corrida 2 (2 a la vez, timeout 90 s): 78 juzgadas, 2 caídas,
  $0,0179 contados en 80 llamadas. **Total contado $0,0387** de un tope de $0,50
  (el dueño autorizó hasta $1). Un timeout no devuelve uso y el proveedor pudo cobrarlo: cota de lo no contado
  $0,0139.
- Estabilidad, el mismo modelo dos veces: de 69 comidas juzgadas en las dos corridas, 53 salen con los mismos tipos de queja (38 de ellas sin ninguna) y 16 cambian. El juez no es determinista; el lote 40 lo tiene que tener en
  cuenta antes de fijar umbrales por código.

## Frontera de la rúbrica (medida, no aplicada)

`cantidad_inconsistente` cubre V4, V6 y V7e; V7a («la lista compra más de lo que los pasos usan») tiene su propia clase,
`lista_de_mas`, que el dueño no usó ni una vez. De los 12 V7a, 3 caen en una comida donde el dueño
marcó `cantidad_inconsistente` del MISMO alimento (p. ej. `2168899312`: «La lista indica 1¼ tomates, pero el paso 1 usa
½ tomate»). Si la rúbrica los uniera, el determinista daría 22/56/25 en vez de
18/60/29. **No se aplica**: la rúbrica es del dueño (y V7a es «residuo» por su decisión del 14-sep).
Queda para él o para el lote 40.

## Lo que NO se hizo

- Inventar el 2.º anotador, adjudicar por el dueño, cambiar detectores (lote 39) o la rúbrica.
- Juzgar con el día entero: el golden set no lo guarda.
- Las cuentas exactas de `recipe_usage` y V8b no aplican al refresco: el caso no guarda la asignación de receta de
  biblioteca ni el formulario.

Test: [`test_p1_plan_lote_60.py`](../tests/test_p1_plan_lote_60.py).
