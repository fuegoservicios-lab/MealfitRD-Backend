# La cuarta guarda de nevera, y el piso que contaba nombres

`[P1-POSTMERGE-WAIVER-SSOT + P1-PANTRY-CONDIMENT-PARITY + P2-SHOPPING-PROTEIN-FLOOR · 2026-08-22]`

Doc canónica. Tests ancla: [`test_p1_postmerge_waiver_ssot.py`](../tests/test_p1_postmerge_waiver_ssot.py),
[`test_p1_pantry_condiment_parity.py`](../tests/test_p1_pantry_condiment_parity.py),
[`test_p2_shopping_protein_floor.py`](../tests/test_p2_shopping_protein_floor.py).
El incidente que las origina está en [`shopping_list_cycle_days.md`](shopping_list_cycle_days.md).

## 1. La cuarta guarda decidía sola

Hay **cuatro** guardas que responden a la misma pregunta («¿puede la nevera bloquear a
este chunk?»):

| # | Dónde | Cuándo | Consultaba la SSOT |
|---|---|---|---|
| 1 | `_should_pause_for_empty_pantry` | pre-pipeline, gratis | sí |
| 2 | gate de reservas (`_res_waiver`) | post-pipeline | sí |
| 3 | validación existencial (`P1-PANTRY-EXIST-WAIVER`) | tras generar | sí (desde 2026-08-08) |
| 4 | **hard-guard post-merge (`P0-4`)** | tras mergear | **no** |

La cuarta leía `_pantry_flexible_mode` a pelo, o sea que honraba **1 de los 4 waivers**.
Medido en el plan real `2245eb45` chunk 2 (`chunk_kind='initial_plan'`, sin flags):

- Las guardas 1 y 3 concedieron `initial_plan_autonomy` y lo dejaron correr.
- La guarda 3 hace `break` en la iteración 0 del bucle de reintentos ⇒ **`attempts = 0`**:
  los 2 reintentos de `CHUNK_PANTRY_MAX_RETRIES` nunca se gastaron y **al LLM jamás se le
  dijo que había un problema de nevera**.
- La guarda 4 no vio el waiver, levantó `_PantryViolationPostMerge` **dentro de
  `conn.transaction()`** → ROLLBACK de los 4 días ya pagados → `pending_user_action`.

Se pagó el LLM entero, no se le dio una sola oportunidad de corregir, y el resultado se
tiró por una condición que las tres guardas anteriores ya habían perdonado.

### Por qué el blanket no lo cazó

`_pantry_gate_waiver_reason` avisa en su docstring: «No añadas una tercera lectura suelta
de `_pantry_flexible_mode`: llama a esta función. `test_p1_pantry_gate_ssot.py` falla si
aparece una guarda que decide sola.»

Ese test recorre una tupla de **un solo elemento**. La promesa del nombre —«ninguna
guarda lee el flag por su cuenta»— era mucho más ancha que lo que miraba. La guarda 3 lo
dice con todas las letras en su propio comentario: «no leía flag alguno, **por eso el
blanket del SSOT nunca la vio**». Un guard que ya no puede fallar es peor que no tenerlo:
da por cubierto lo que no cubre.

`test_p1_postmerge_waiver_ssot.py` declara ahora el **inventario explícito** de lecturas
dentro de `_chunk_worker` (`LECTURAS_DECLARADAS`). Ninguna nueva puede aparecer sin que
alguien la justifique ahí.

> Nota de método: la primera versión del helper que extrae el bloque a inspeccionar
> tomaba el rango entre la primera y la última línea con prefijo `_p04_`. Como
> `_p04_pause_snap` vive ~2.000 líneas más abajo, el «bloque» abarcaba media función y la
> aserción pasaba por encontrar una llamada de **otra** guarda. El test que arreglaba un
> guard que no podía fallar nació sin poder fallar. Ahora ancla por tooltip.

### El contrato nuevo

Con waiver activo, el post-merge **anota y entrega**: marca las comidas ofensoras
(`_pantry_violated`, helper `_mark_meals_violating_pantry`) y deja que el bloque se
publique. Es el mismo contrato que el flexible ya tenía («la entrega marcada es el
contrato») y que el camino síncrono del chunk 1 aplica desde P0-5. Un menú con lo que
falta señalado es estrictamente mejor que un Dashboard vacío.

### `_p0_4_violations` era escritura muerta

Un grep sobre todo `backend/` devolvía **una** aparición: la escritura. El detalle de por
qué murió el chunk se guardaba y se descartaba, así que al reanudar el modelo recibía el
mismo prompt que ya había fallado. Ahora `_resolve_pantry_pause_markers` —el único punto
por el que pasan las 7 rutas de reanudación— lo promueve a
`form_data["_pantry_correction"]`, que es lo que `build_pantry_correction_context`
convierte en el bloque «CORRECCIÓN OBLIGATORIA». No pisa una corrección ya presente: la
del worker es más fresca.

## 2. El prompt autorizaba lo que el validador castigaba

`build_pantry_correction_context` le promete al modelo: «Condimentos básicos (sal,
pimienta, aceite, ajo, **cebolla**, cilantro) están siempre permitidos». El prompt del
catálogo añade «comino, cúrcuma, laurel, tomillo, curry, cebolla en polvo» y una
excepción de repostería: «SÍ puedes usar polvo de hornear, levadura, bicarbonato y
vainilla … aunque no estén en la lista».

`constants._ALLOWED_CONDIMENTS` tenía **once** palabras y no incluía ninguna. El bloque
del incidente murió con «INEXISTENTES: ¼ cdta de polvo de hornear, ½ cdta de comino,
1 cebolla, ½ hoja de laurel» — las cuatro autorizadas por escrito. **El sistema castigaba
al modelo por obedecerle.**

Se arregla del lado del validador porque el prompt es el contrato ofrecido: si el sistema
promete algo y luego lo penaliza, el fallo está en quien juzga. La paridad la ancla
`test_p1_pantry_condiment_parity.py` con la tabla `AUTORIZADOS_POR_EL_PROMPT`, que
registra de qué prompt sale cada permiso.

⚠️ **No fusiones `_ALLOWED_CONDIMENTS` con `culinary_coherence.CONDIMENT_EXEMPT`.** Se
parecen y no son lo mismo: la primera responde «¿tiene que existir en la nevera?», la
segunda «¿necesita método de cocción?». Colapsarlas sería escribir la cuarta tabla que
`P1-DIET-CANON-SSOT` prohíbe, y por la peor razón: que se parecen.

Exentar un condimento del gate **no** lo borra de la lista de compras: el agregador lo
sigue costeando. La exención sólo significa «no te niegues a cocinar porque falte».

## 3. El piso contaba nombres, no comida

`_shopping_list_completeness` medía nombres distintos contra un mínimo escalado (12 para
30 días) y era ciega a la categoría. La lista del incidente:

| Categoría | Ítems |
|---|---|
| Despensa | 12 |
| Lácteos | 4 |
| Vegetales | 4 |
| Frutas | 2 |
| **Proteínas** | **1** (Huevo) |
| Víveres | 1 (Papa) |

25 ≥ 12 ⇒ ni `is_empty` ni `is_sparse`. El sistema sabía contar alimentos pero no sabía
preguntar «¿con esto se puede comer?».

Ahora expone `distinct_proteins` e `is_protein_starved` (knob
`MEALFIT_SHOPPING_MIN_PROTEINS`, default 2). **Los Lácteos no cuentan**: la lista del
incidente tenía 4 y, si contaran, el caso que este piso existe para cazar habría pasado
igual.

### El veredicto caducaba

`_shopping_completeness` se calculaba SOLO en `assemble_plan_node`.
`grep _shopping_completeness backend/routers/plans.py` daba **0 matches**: el recálculo
reescribía las 4 listas y jamás re-medía, así que el plan quedó persistido afirmando
`distinct: 49` mientras publicaba 25. Un veredicto que describe una lista que ya no
existe es peor que ninguno, porque un operador lo cree. Ahora se re-mide en cada
recálculo.

**Mide y avisa; no bloquea.** Un rechazo por falta de proteínas rompería dietas
legítimamente poco variadas, y el modo de fallo que nos ocupa no era «se generó mal» sino
«la lista se erosionó después» — eso lo cierra `P0-SHOPPING-CYCLE-DAYS`.

## Knobs

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_SHOPPING_MIN_PROTEINS` | `2` | proteínas distintas por debajo de las cuales la lista se marca `is_protein_starved` |
| `MEALFIT_INITIAL_CHUNK_PANTRY_AUTONOMY` | `True` | (pre-existente) concede `initial_plan_autonomy`; ahora también lo honra el post-merge |

## Lo que queda abierto

- **No hay umbral de cantidad mínima** en el gate: «7 g de arroz blanco crudo» bloquea
  igual que 200 g de pollo. Se dejó fuera a propósito — subir un umbral cambia la
  semántica de seguridad para alimentos reales, y con el waiver honrado los
  `initial_plan` ya no mueren ahí. Sigue vivo para `rolling_refill` sin waiver.
- **`malla` no existe en `_to_base_unit`**: «2 mallas de Papa» (≈10-12 papas) se cuenta
  como 2 unidades, y por eso «3 papas medianas» excedía el inventario del incidente.
- **Las otras 4 lecturas del flag** dentro de `_chunk_worker` siguen sueltas; están
  declaradas en `LECTURAS_DECLARADAS` con su motivo, y ninguna decide una pausa hoy.
