# El día determinista — qué es, qué cubre y cómo se enciende

`[P1-DETERMINISTIC-DAY · 2026-09-08]` · motor: [`backend/deterministic_day.py`](../deterministic_day.py)

Arma un día de plan **sin llamar al modelo**: elige el plato del Dish Registry puntuando por
macros, escala los gramos al objetivo calórico, inclina proteína contra carbohidrato dentro de un
tope y pega la receta congelada de la biblioteca de 140. Si cualquier paso no cuadra devuelve
`None` y el pipeline cae al camino de siempre — la generación **no** depende de que esto acierte.

---

## Los knobs: son DOS, y uno solo no hace nada

| Knob | Default | Qué hace |
|---|---|---|
| `MEALFIT_DETERMINISTIC_DAY` | `False` | Enciende el día determinista **para todos**. |
| `MEALFIT_DETERMINISTIC_DAY_USERS` | *(vacío)* | Lista de uuids separados por coma. Con el knob global **apagado**, sólo esos usuarios reciben días deterministas. Es la vía de encendido segura. |
| `MEALFIT_RECIPE_LIBRARY_SELECT` | `False` | Knob maestro de la biblioteca de recetas. **Sin él, el día determinista no construye NADA.** |
| `MEALFIT_DETERMINISTIC_DAY_CANDIDATES` | `25` | Candidatos por franja, clamp `[3, 60]`. |

> ⚠️ **`MEALFIT_DETERMINISTIC_DAY` solo no hace nada.** `construir_comida` devuelve `None` cuando
> no hay receta congelada —a propósito: sin ella no hay determinismo del texto— y
> `recipe_for_dish_name` empieza con `if not library_select_enabled(): return None`. Encender uno
> sin el otro da **0 días construidos** y parece que la feature no sirve. Es un rato perdido que ya
> pagué yo; queda anclado en `test_p1_deterministic_day_canary.py`.

### Los tres estados del canario

| `MEALFIT_DETERMINISTIC_DAY` | `MEALFIT_DETERMINISTIC_DAY_USERS` | Quién recibe días deterministas |
|---|---|---|
| off | vacío | **nadie** (estado actual en producción) |
| off | `<uuid>` | **sólo ese usuario** |
| on | *(indiferente)* | **todos**, invitados incluidos |

Una lista vacía cierra, nunca abre — el modo de fallo clásico del split por env var (`"".split(",")`
da `[""]`, que casaría con un `user_id` vacío) está cubierto por test.

### La identidad sale de `form_data["user_id"]`

Medido, no supuesto: en `pipeline_metrics`, `generate_day_1` trae `user_id` en **151 de 158**
filas (las 7 sin él son invitados). El cron lo estampa explícito para los bloques 2+.

*(Casi lo doy por inerte con una sonda rota: conté planes con `user_id` dentro de
`plan_data->'form_data'` y salió 0 de 97 — pero **`plan_data` no persiste `form_data` en ningún
plan**, así que ese 0 no medía nada.)*

---

## SOP de encendido

El `.env` de producción vive en el VPS (`/opt/mealfit/backend/.env`) y **el deploy no lo sube**
(`--exclude="backend/.env"`), así que el knob se cambia allí y se reinicia el servicio.

1. Encender **para un solo usuario** — en el VPS, sin tocar el knob global:
   ```
   MEALFIT_RECIPE_LIBRARY_SELECT=1
   MEALFIT_DETERMINISTIC_DAY_USERS=<uuid-del-dueño>
   ```
2. Reiniciar el backend y confirmar que arrancó: `curl https://app.bioboros.com/health/version`.
3. Generar un plan con esa cuenta y **contar** (ver «Telemetría» abajo). El rastro es
   `_meal_source="deterministic"` en cada comida y `_recipe_source="library"`.
4. Revisar los días servidos. Si algo no convence: borrar la línea `..._USERS` y reiniciar —
   marcha atrás sin despliegue.
5. Sólo después, y con la medición delante, considerar `MEALFIT_DETERMINISTIC_DAY=1`.

---

## Cobertura medida (2026-09-08)

Barrido de 14 días × 3 perfiles clínicos (2.000 kcal · 150 g proteína · 200 g carbos · 60 g grasa),
dos corridas independientes:

| Métrica | Resultado |
|---|---|
| Días construidos sin LLM | **33 de 42** con esqueletos coherentes |
| Idénticos byte a byte entre corridas | **33 de 33** |
| Error calórico | **−0,3 %** (máx. −0,4 %) |
| Proteína dentro de ±15 % | 19 de 42 · media **−18,6 %** |
| Carbohidratos dentro de ±15 % | 16 de 42 · media **+16,9 %** |
| Comidas sucias (escáner culinario + backstop clínico) | **0 de 132** |

### Lo que NO cubre, y por qué

- **Cenas de res y pavo.** El registry tiene 3 plantillas de res y 1 de pavo para cena, y están
  escritas ligeras (313–356 kcal) frente a un objetivo de 600. Necesitarían factor 1,68–1,92 y el
  techo de la franja es 1,60. **No se ensancha la banda para mejorar el número**: servir casi el
  doble de la porción curada es el modo de fallo de `P1-CULINARY-V6-STEP-OVERASK` (la receta dice
  «una pechuga» y la lista pide 1,9). Se arregla escribiendo cenas de res/pavo más contundentes —
  es un hueco de **datos**, no de código, y la biblioteca la juzga el dueño a ciegas.
- **Sesgo de macros.** Proteína baja y carbos altos de forma sistemática en los tres perfiles.
  **Subir el tope de inclinación NO es la palanca, y está medido**: barrido de `_TILT_TOPE` a
  0,35 · 0,50 · 0,70 · 1,00 · 1,50 (más de 4×) mueve la proteína media de **−18,6 % a −18,0 %** —
  o sea, nada. La restricción que ata es el suelo de los carbohidratos, y por debajo de eso el
  plato simplemente no tiene más proteína que repartir. El sesgo es de **datos** (qué platos hay y
  con qué densidad proteica), no de ajuste; subir el tope sólo distorsionaría las porciones a
  cambio de cero. Queda abierto: las palancas reales son el **scorer** de `elegir_plantilla` y la
  composición de la biblioteca.
- **Las 4 plantillas sin receta** (Menta, Salami de pavo, Chillo, Zapote) son
  `declared_unresolved` a propósito. Dos tienen sustituto plausible en el catálogo —Zapote ≈
  Níspero, Chillo ≈ filete de pescado blanco— pero cambiar un pescado por otro **cambia el plato**
  y eso lo decide el dueño.

Un día que no construye devuelve `None` y lo genera el modelo: el estado de siempre.

---

## Memoria entre días (P1-PLAN-LOTE-2 · 2026-09-11)

Cada día se armaba sin saber qué comieron los anteriores. Medido el 09-10 sobre los 30 días del dueño:
el tope de repetición exacta de la política (`balanced`: 2 veces por 7 días) se rompía en **9 ventanas**
con las puertas de variedad del día apagadas y en **28** con ellas encendidas. Ahora
`generate_days_parallel_node` comparte una lista (`_det_prev`) entre los días del run —las tareas se
crean en orden y este módulo corre en la primera fase síncrona de cada una, así que el día N ve a los
N-1— y `build_day_for_skeleton(..., memoria=)` la **lee y la actualiza**. Si el bloque continúa un plan
(`days_offset` > 0), los días ya entregados se cargan **una vez** de la base y entran al principio.
`_conteo_ventana` cuenta cuántas veces se sirvió cada plantilla en los 6 días anteriores (por
`_template_id` o, para días del modelo, por nombre exacto contra el registry) y `elegir_plantillas`
manda al **final** —no fuera— a las que agotaron su cuota (`_max_repeticion_7d`, de
`horizon.repetition_limits_for`). La rotación gira sólo sobre los frescos. Quedarse sin plato por no
repetir sigue siendo peor que repetir: si todos están saturados, se sirve igual.

## Seguridad: el backstop no es opcional

Un día que sale de aquí **sí pasa por `assemble_plan_node` y `review_plan_node`** (las aristas del grafo
son incondicionales — la versión anterior de este párrafo afirmaba lo contrario, y esa premisa falsa
vivió en cuatro docstrings y dos commits hasta P1-AUDITORIA-ARQ-VERIFICADA). Lo que NO tiene es un
segundo juicio antes de entrar en el plan: por eso `verifica_comida` corre las seis capas del escáner
culinario **y** `clinical_backstop_for_meal` sobre cada plato antes de devolver el día, y una violación
**rechaza** el día entero (cae al LLM), no sólo se registra. Es la misma clase de superficie que
`P0-DEGRADED-SAFETY-SCAN` cerró para el path degradado, que ése sí bypasea el ensamblador.

El backstop ya cazó un fallo real: el selector llamaba a `template_candidates` **sin pasarle las
alergias**, y un desayuno con huevo llegó a un alérgico al huevo. Se cerró pasando
`exclude_allergens` y `diet` — *una última línea de defensa que trabaja sola dejó de ser defensa en
profundidad*.

---

## Telemetría

Cada comida determinista lleva `_meal_source="deterministic"`, `_template_id` y `_scale_factor`;
la receta congelada añade `_recipe_source="library"`. Para contar la adopción en producción:

```sql
SELECT
  count(*) FILTER (WHERE m->>'_meal_source' = 'deterministic') AS deterministas,
  count(*)                                                     AS total_comidas,
  count(*) FILTER (WHERE m->>'_recipe_source' = 'library')     AS con_receta_congelada
FROM public.meal_plans p,
     LATERAL jsonb_array_elements(p.plan_data->'days')  AS d,
     LATERAL jsonb_array_elements(d->'meals')           AS m
WHERE p.created_at > now() - interval '7 days';
```

Sin esta consulta, encender el knob y no encenderlo se ven **idénticos** desde fuera.

---

## Tests

- [`test_p1_deterministic_day.py`](../tests/test_p1_deterministic_day.py) — el algoritmo.
- [`test_p1_deterministic_day_wired.py`](../tests/test_p1_deterministic_day_wired.py) — el enganche
  al pipeline y el techo del god-file.
- [`test_p1_deterministic_day_backstop.py`](../tests/test_p1_deterministic_day_backstop.py) — las
  dos capas de verificación.
- [`test_p1_deterministic_day_canary.py`](../tests/test_p1_deterministic_day_canary.py) — el
  canario, la lista vacía y los dos knobs.
