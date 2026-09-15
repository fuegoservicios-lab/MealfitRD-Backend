# Plan ejecutable para otro agente — lotes 38 a 44 · 2026-09-14

Lo que queda del [plan de pendientes](plan_pendientes_2026_09_11.md) que puede cerrar código, ahora que el dueño entregó la
**anotación con rúbrica de los 80 casos** del golden set (Hoja del dueño, 2026-09-13). Este documento está escrito para un
agente **sin el contexto de las sesiones anteriores**: trae el protocolo, las cifras medidas hoy, los ficheros exactos y el
criterio de «terminado» de cada lote. Léelo entero antes de tocar nada. Si algo aquí contradice el código, manda el código
y se anota la discrepancia en el informe. El protocolo largo (entorno, gate, checklist de cierre) es el de
[`plan_agente_lotes_32_36_2026_09_13.md`](plan_agente_lotes_32_36_2026_09_13.md) §0; aquí va resumido.

| Lote | Ítem del plan | Qué se entrega | Tamaño |
|---|---|---|---|
| 38 | C1 (cierre) · C0 | ✅ HECHO (`P1-PLAN-LOTE-60` · 2026-09-15): adjudicador arreglado, columnas de la máquina refrescadas, línea base ESTRICTA publicada — determinista 11/30/36 → 18/60/29 (tp/fp/fn; recall 23,4 → 38,3 %, precisión 26,8 → 23,1 %), juez 2/32/7 → 0/28/9. Ver la sección «Lote 38» | — |
| 39 | C5 / CUL-P1-05 | Detector para `coccion_faltante` (8 high sin ningún código) y recall de `seco_sin_coccion` (10 high, 0 detectados) y `usa_lo_que_no_esta` (1 de 9) | 1 sesión |
| 40 | C5 / C6 (parte medible) | Precisión del juez por código sobre la verdad humana; códigos con precisión baja pasan a observación; recomendación para el flip de C6 (que decide el dueño) | ½-1 sesión |
| 41 | Tarea propuesta 13-sep | La tormenta de reintentos del catálogo con la base caída: caché negativa corta en `get_master_ingredients` y `catalog_capability` | ½ sesión |
| 42 | E1 (operativo) | Retirar los worktrees viejos que estén limpios y ya mergeados; los demás, listados para el dueño (G4) | ¼ sesión |
| 43 | B6 · E5 · D7 · B9 | Re-mediciones con FECHA: los tres scripts del lote 36 el **2026-10-10**; el embudo B9 a las 2 semanas del lanzamiento | ¼ sesión, en su fecha |
| 44 | Decisiones del dueño 14-sep | Aplicar lo delegado: 3 recetas a la biblioteca, claras en botella ≥ 4, cohorte B de la lista canónica = canario, pesos 2.0/1.0 en el canario, `block` en PayPal | 1-2 sesiones |
| 49 | Prueba RD 5 del dueño 14-sep (fuera del plan) | ✅ HECHO en otra sesión (`P1-PLAN-LOTE-49`): el ingrediente que da nombre sube a su piso al final si el día cabe; el cerrador busca un lácteo que el día no tenga en desayuno y merienda, no añade curados y deja su paso en su sitio; el tope de huevos reescribe la compra; los planes recientes cuentan sólo planes con días; el tiempo es falta del armador; texto de latas y pasos breves. Ver la sección «Lote 49» | — |
| 48 | Prueba RD 4 del dueño 14-sep (fuera del plan) | ✅ HECHO en otra sesión (`P1-PLAN-LOTE-48`): los cerradores escalan la base propia en vez de colgar otra, en un jugo el lácteo va al lado, el «queso» a secas toma el nombre del plato, el desalado se quita como cláusula y la pulpa tiene techo; el armador prefiere el plato que ya llega al piso de proteína (por debajo de la cuota de repetición) y la re-elección re-mide la cola. Ver la sección «Lote 48» | — |
| 47 | Prueba RD 3 del dueño 14-sep (fuera del plan) | ✅ HECHO en otra sesión (`P1-PLAN-LOTE-47`): la autocrítica y la regeneración re-eligen un día determinista en vez de reescribirlo con el LLM, lo que el corrector deja igual vuelve con su procedencia, el armador conoce las reglas fijas de la autocrítica; parches de tiempo, casabe y queso. Ver la sección «Lote 47» | — |
| 46 | Prueba RD 2 del dueño 14-sep (fuera del plan) | ✅ HECHO en otra sesión (`P1-PLAN-LOTE-46`): repetición entre planes y tope del bloque en el día determinista, el revisor no reintenta lo determinista, el ingrediente que da nombre al plato no se pierde, parches de pasos. Ver la sección «Lote 46» | — |
| 45 | Prueba RD del dueño 14-sep (fuera del plan) | ✅ HECHO en otra sesión (`P1-PLAN-LOTE-45`): receta de biblioteca en su orden, día determinista con la familia del blueprint y el tiempo de cocina, sin reintento inútil. Ver la sección «Lote 45» | — |

Orden recomendado: 38 → **44** → 39 → 40 (el 44 cambia lo que come el dueño, que quiere probar la generación ya; 38, 39 y 40 comparten instrumento y verdad humana), luego 42. El 41 lo está ejecutando otra sesión desde el 14-sep. 43
sólo en su fecha. Si uno se atasca, se cierra lo que haya con su «lo que NO se hizo» y se sigue con el siguiente.

---

## 0. Cómo se trabaja aquí (resumen; el detalle está en el plan 32-36 §0)

### 0.1 Entorno

- Backend: `C:\Users\angel\OneDrive\Escritorio\Nodalia\MealfitRD\Software\MealfitRD.IA\backend` (repo propio, `origin/main`). El
  workspace raíz (`CLAUDE.md`, `docs/`, espejo `migrations/`) y `frontend/` son repos hermanos.
- Python: `%USERPROFILE%\miniconda3\envs\mealfit\python.exe`, siempre con `PYTHONIOENCODING=utf-8 PYTHONUTF8=1 PYTHONHASHSEED=0`.
  El env `mealfit` está alineado con `requirements.txt` desde el 2026-09-13.
- Temporales y scripts de un solo uso: en el **scratchpad de la sesión**, nunca en el repo. Ficheros grandes con la herramienta
  Write (los heredocs de Git Bash rompen barras invertidas); los parches a ficheros de producción con un script Python que
  `assert`e que su ancla aparece exactamente una vez.
- Memoria del proyecto (fuera del repo): `C:\Users\angel\.claude\projects\C--Users-angel-OneDrive-Escritorio-Nodalia-MealfitRD-Software-MealfitRD-IA\memory\`.
- **Un solo gate a la vez** en esta máquina (16 GB): dos sesiones de pytest con `-n` se matan entre sí (MemoryError, workers
  caídos, `tmp_path` borrado) y producen falsos rojos. Réplicas del árbol fuera del scratchpad (MAX_PATH).

### 0.2 Prohibiciones

- **Secretos**: nunca crear, leer en claro ni pegar. Se usan vía `os.environ`.
- **VPS**: no tocar `.env` ni systemd. El deploy (`deploy-mealfit.ps1`) es el único camino a producción y **empaqueta el árbol de
  trabajo**, no `main`: `git status` limpio antes de desplegar.
- **Base de producción (Neon)**: scripts con `conn.read_only = True` y sólo SELECT. Cambios de datos, sólo por migración SSOT
  (`migrations/` + `backend/migrations/` idénticos, `scripts/apply_migration.py`). Ningún bench escribe telemetría en producción.
- **Árbol del dueño**: `10k-websites/` (untracked en la raíz) es suyo. Jamás `cp` de un fichero entero sobre el árbol. Commits
  **con rutas explícitas**, nunca `-A`. Mientras corre el gate no se edita ningún fichero del backend.
- **Topes de los god-files** (`tests/test_p3_shopping_projection_pkg.py:49-52`), «extraer, no subir el tope»: `graph_orchestrator.py`
  52.600 (hoy 51.956), `cron_tasks.py` 36.550 (35.851), `routers/plans.py` 18.100 (17.743), `shopping_calculator.py` 14.400
  (14.174), `plan_jobs.py` 800 (714).
- **Tests**: sin los literales `psycopg.connect(`, `connection_pool.open(`, `load_dotenv(` (la regla `_LOCAL_ONLY` de
  `tests/conftest.py` salta el módulo entero sin base y el test desaparece de la CI en silencio); sin `"CLAUDE.md"`, `runbook_`,
  `".env"`. Los scripts se cargan con `importlib.util.spec_from_file_location`, no con `sys.path.insert`.
- **Decisiones del dueño** (no implementar aunque parezcan obvias): V7a «el paso pide MENOS», flip de `MEALFIT_CULINARY_JUDGE_GUARD`
  (C6), fase B de E5 (cohorte), encender `MEALFIT_GUARD_UNDERSUPPLY_SEVERE` o `MEALFIT_TRIP_WINDOWED_PERISHABLES`, fusionar filas
  sinónimas del catálogo, cablear `MEALFIT_RECIPE_REPAIR_BUDGET`. Descargar PDFs de tablas nacionales de composición: pedir permiso.
- **Gasto LLM**: sólo corridas dirigidas y con tope. El único gasto previsto aquí es el refresco del juez del lote 38, bajo el
  mismo tope que el bench del lote 30 (**$0,50**, contado en proceso).

### 0.3 El gate (mismo contrato que la CI) y el cierre de un lote

Tres fases, en este orden, desde `backend/`: (A) `pytest tests/ -q --tb=short -m "not e2e" -n 2 --dist loadfile
--max-worker-restart=4 --ignore <cada fichero de QUARANTINE de ci.yml>`; (B) la cuarentena en serie con `-p no:cacheprovider`;
(C) `python scripts/prod_profile_gate.py --workers 3` (perfil de producción + batería). Hoy: A 24.781 verdes, B 107, C 23.787 + 29.
Usa `--basetemp` privado (`C:\tmp\pt_<lote>`).

Cierre de cada lote, todo y en este orden: tests nuevos en `tests/test_p1_plan_lote_<N>.py` · marker `_LAST_KNOWN_PFIX =
"P1-PLAN-LOTE-<N> · YYYY-MM-DD"` en `app.py` (13 tests de lotes anteriores exigen `N` ≥ el suyo; `test_p2_hist_audit_14_marker_test_link`
exige el fichero de test; `test_p3_marker_memory_xlink` exige el marker LITERAL en `MEMORY.md`) · docs (`knobs_reference.md` si hay knob,
fila del ítem en `plan_pendientes_2026_09_11.md`, el doc del área) · párrafo del lote en la memoria
`project_plan_pendientes_lotes_2026_09_11.md` + una línea en `MEMORY.md` con el marker · gate 3 fases en verde · commit con rutas
explícitas y mensaje que cuente lo medido · push · deploy (`pwsh -NoProfile -File <raíz>/deploy-mealfit.ps1 backend -SkipTests`) ·
`curl -s https://bioboros.com/health/version` → `last_known_pfix` = tu marker y `drift: false` · CI verde (run de GitHub; las dos patas
cierran hoy en ~10 min con techo de 30). Un lote que sólo toca docs/scripts de medición no bumpea marker ni despliega.

**Numeración (P1-PLAN-LOTE-45 · 14-sep).** El marker nunca baja: cada test de lote exige `N` ≥ el suyo. El 41 y el 45 ya
están cerrados y desplegados, y del 46 al 49 también, así que los lotes pendientes de este plan (38, 39, 40, 42, 43,
44) se cierran con el **siguiente número libre** (≥ 56 al escribirlo; el 38 cerró como `P1-PLAN-LOTE-60` el 15-sep, así que hoy ≥ 61) —marker, `tests/test_p1_plan_lote_<N>.py` y memoria con ese número— y el título del commit
cita su número de plan («`P1-PLAN-LOTE-56` (lote 38 del plan)»). Cerrar con el número del plan haría fallar los tests 41 y 45 a 53.
El 50, el 51 y el 53 los tomó la sesión del coach (14-15 sep, `P0-CHAT-IDENTITY-FROM-TOKEN`, `P1-DIARY-CLAIM-PERFECTIVE` y la
batería de escritura del coach y los escáneres), ajenos a este plan; el 54 (panel del tiempo de cocina) y el 55 (minutos de los pasos del registry) están reservados.

---

## Lote 38 · C1 (cierre): la anotación del dueño entra al instrumento

**Estado: ✅ HECHO · `P1-PLAN-LOTE-60` · 2026-09-15.** determinista 11/30/36 → 18/60/29 (tp/fp/fn; recall 23,4 → 38,3 %, precisión 26,8 → 23,1 %), juez 2/32/7 → 0/28/9; detalle, tablas por clase y coste en
[`culinary_baseline_estricto_2026-09-15.md`](culinary_baseline_estricto_2026-09-15.md). Discrepancias con lo escrito abajo, anotadas:
(1) «nombra algún alimento» se decide con la lista de ingredientes de la comida, no con las comillas (las tres formas dan
los mismos aciertos con las columnas del 09-06; ésta es la única que ve un alimento sin comillas); (2) el refresco escribe
también el alimento que el detector acusa, porque el detalle de V6/V7e cita el paso y nombra otros; (3) el juez se llamó
con una comida por llamada (el golden set no guarda el día) y dos veces: la primera corrida perdió 9 casos por timeout y no
llevaba el componente; (4) V7a frente a `cantidad_inconsistente` es una frontera de la RÚBRICA —medida, no aplicada—.

### Qué hay (medido 2026-09-14)

- **La anotación**: `docs/culinary_golden_anotaciones_angelo.json` (volcado tal cual de la Hoja del dueño, anotación CIEGA: lo que marcó
  la máquina se mostraba sólo después de decidir). 80/80 casos: **66 defecto · 5 dudoso · 9 ok**; 65 defectos con clase de la
  RUBRICA. Por clase (high/minor): `cantidad_inconsistente` 19 (12/7), `seco_sin_coccion` 10 (10/0), `usa_lo_que_no_esta` 9 (4/5),
  `coccion_faltante` 8 (8/0), `paso_incoherente` 7 (3/4), `ingrediente_huerfano` 5 (3/2), `nombre_no_corresponde` 2 (0/2), `otro` 2
  (0/2), `verbo_alimento` 1, `masa_sobrante` 1, `estructura_del_plato` 1 (high). Un caso, **`060b4fda6a`**, está «defecto» con la
  lista de defectos vacía. Consistencia con la etiqueta binaria del 2026-09-07 del mismo anotador: **78/80 iguales** (2 «defecto» →
  «dudoso»).
- **El instrumento da 0 aciertos** con esa anotación: `python scripts/culinary_golden_score.py --estricto --anotaciones
  docs/culinary_golden_anotaciones_angelo.json` → determinista tp=0 fp=41 fn=46, juez tp=0 fp=34 fn=9. Causa, en
  `scripts/culinary_golden_score.py::_adjudicar_hallazgos` (~línea 321): si el defecto trae `alimento`, exige que sea **subcadena del
  texto del hallazgo**; el dueño rellenó `alimento` en casi todos («Carne de res magra», «Casabe y queso fresco.», «Sandía, semillas de
  girasol y…») y los textos de la máquina no nombran alimento (`V4: ingrediente declara 85 g, pasos declaran 140 g` es EXACTAMENTE el
  defecto del caso `0108f857ae`, «Lista: 85 g de res; paso 1: porción de 140 g», y sale FN + FP). Sin ese filtro, la misma anotación
  da: determinista **tp=11 fp=30 fn=35** (precisión 26,8, recall 23,9), juez **tp=4 fp=30 fn=5** (11,8 / 44,4); por clase
  `cantidad_inconsistente` 7/0/12, `ingrediente_huerfano` 2/0/3, `paso_incoherente` 3/0/4, `nombre_no_corresponde` 1/0/1,
  `usa_lo_que_no_esta` 1/0/8, `verbo_alimento` 1/0/0, `seco_sin_coccion` 0/0/10, `estructura_del_plato` 0/0/1, `masa_sobrante` 0/0/1;
  sin ningún código que los mecanice: `coccion_faltante` 8, `otro` 2. *Un veredicto vale lo que valga el instrumento que lo produjo.*
- **Las columnas de la máquina son viejas**: `docs/culinary_golden_set.json` tiene `generado: 2026-09-06` y sus `maquina_determinista`
  sólo traen **V1 12 · V2 2 · V3 16 · V4 12 · V5 2** — ni un V6, V7a-e, V8, V9 (nacieron en los lotes 22-31, del 09-12), y
  `maquina_juez` (combo_absurdo 5, nombre_no_corresponde 8, paso_incoherente 12, slot_inapropiado 4, tecnica_impropia 7) es anterior
  al juez con estado `[dudosa]` del lote 28. Medir el instrumento de hoy contra esas columnas es medir el de hace ocho días.
- Hay UN anotador: `acuerdo.anotadores=1`, `kappa=None`, `promocion_habilitada=False` (exige 2). El 2.º anotador es del dueño.

### Qué hacer

1. **Adjudicador**: en `_adjudicar_hallazgos`, el `alimento` del defecto sólo restringe cuando el texto del hallazgo **nombra algún
   alimento** (V3 y V5 lo traen entre comillas; V4 no). Compara por tokens normalizados (`_sin_acentos`, minúsculas, singular) y
   acepta si CUALQUIER token ≥ 4 letras del `alimento` aparece en el texto; si el hallazgo no trae alimento, empareja sólo por código.
   Anota en el detalle `emparejado_por: codigo|codigo+alimento`. Test: el caso `0108f857ae` da tp=1 con el `alimento` del dueño puesto.
2. **Caso sin defecto listado**: «defecto» con `defectos == []` no es `completo`: estado `sin_rubrica` (no puntúa) y el informe lo
   nombra. Hoy es `060b4fda6a`; queda para que el dueño lo complete en la hoja.
3. **Refrescar la máquina**: un script de solo lectura (`scripts/culinary_golden_refresh.py`) que, para cada caso del golden set, corre
   el scan determinista ACTUAL sobre `ingredientes` + `pasos` (mira cómo `scripts/culinary_baseline.py::_medir_filas` materializa una
   comida y qué función de `culinary_coherence` llama; el builder `build_culinary_golden_set.py` leyó los hallazgos PERSISTIDOS de 96
   planes, no volvió a escanear) y escribe `maquina_determinista_<fecha>` SIN borrar la columna del 09-06 (la comparación
   antes/después es el resultado). El juez: `--con-juez` opcional, mismo modelo que producción, tope **$0,50** contado en proceso,
   0 escrituras en la base; si el tope se alcanza, se para y se dice cuántos casos quedaron. Sin `--con-juez` la columna del juez se
   queda como está y el informe lo declara.
4. **Línea base estricta**: `docs/culinary_baseline_estricto_2026-09-<dd>.md` con las dos tablas (antes/después del refresco) —
   precisión/recall por capa con IC95, por clase, no mecanizable, acuerdo — y el JSON al lado (`--json`). Es la cifra contra la que
   se miden los lotes 39 y 40: **no se toca después**.
5. Test `tests/test_p1_plan_lote_38.py`: (a) el adjudicador con el ejemplo real; (b) `sin_rubrica` para defecto sin defectos; (c) el
   fichero del dueño carga con `cargar_anotaciones` y tiene 80 casos con veredicto no vacío; (d) el refresco NO borra la columna vieja
   (parser-based sobre el script); (e) docs + marker ≥ 38.

### Criterio de terminado

- El estricto con la anotación del dueño deja de dar 0 tp por construcción, y el informe dice cuántos emparejamientos fueron por
  código y cuántos por código+alimento.
- Golden set con la columna refrescada y la del 09-06 conservadas; la línea base estricta publicada con fecha, antes y después.
- Ni una escritura en la base; gasto del juez ≤ $0,50 y contado.
- Lo que NO se hace aquí: inventar el 2.º anotador, adjudicar por el dueño, cambiar detectores (eso es el lote 39).

---

## Lote 39 · Detectores para lo que el dueño más marcó y la máquina no ve

### Medido

- `coccion_faltante` es la **3.ª clase más frecuente y toda high** (8 de 65) y en la RUBRICA su conjunto de códigos es `set()`: ningún
  detector, ni determinista ni juez, puede acertarla. `seco_sin_coccion` (10, toda high) mapea a **V7c**, que no aparece ni una vez en
  las columnas del golden set (verificar tras el refresco del lote 38: si V7c sigue en 0 sobre esos 10 casos, el detector no cubre lo
  que el dueño llama así). Evidencias típicas del dueño: «La lista incluye 40 g de arroz blanco crudo y el paso 2 pide incorporarlo
  directamente, sin hervirlo» (`01606d0573`); «lentejas y habichuelas negras secas … el paso 2 solo las calienta en la plancha durante 4
  y 2 minutos, sin cocción previa» (`01c22c6847`).
- El juez SÍ vio dos de esos casos pero con OTRO código: `slot_inapropiado: El desayuno incluye 40 g de arroz blanco crudo que se
  'incorpora'…` y `tecnica_impropia: El ingrediente es '½ taza de lentejas secas' pero la receta solo las 'calienta'…`. Con la RUBRICA
  estricta cuentan como FP + FN a la vez. Eso es del lote 40 (mapa de códigos), no de éste.
- `usa_lo_que_no_esta` (V5): 9 del dueño, 1 acierto, V5 sólo disparó 2 veces en 80 casos. Ejemplo del dueño: «Los pasos 1 y 3 piden
  almendras fileteadas, pero no aparecen en la lista» (`00fae1c079`).

### Qué hacer (medir antes, detector después, warn siempre)

1. Sobre los 8 `coccion_faltante` y los 10 `seco_sin_coccion` del dueño, escribe primero a mano qué señal los separa de los 9 `ok`
   (crudo/seco en la LISTA — `master_ingredients.prep_methods` / la metadata culinaria de `culinary_coherence` — y NINGÚN verbo de cocción
   sobre ese alimento en los PASOS, o sólo «calentar»/«incorporar»). Cuenta cuántos de los 18 la cumplen y cuántos de los 9 `ok` también
   (falsos positivos esperables). Ese conteo va al informe ANTES de escribir código.
2. Detector **V7f `coccion_faltante`** en `culinary_coherence` (capa 1, `warn`, knob `MEALFIT_CULINARY_V7F` default `true` en la suite,
   registrado en `_KNOBS_REGISTRY`), texto del hallazgo con el ALIMENTO nombrado (para que el adjudicador del lote 38 lo empareje), y la
   RUBRICA gana `"coccion_faltante": {"V7f"}`. Revisa V7c: si sus 0 disparos son porque exige una forma («seco» en la línea) que estos
   casos no tienen, amplía la forma, no la severidad. Para V5 mide por qué no dispara con «almendras fileteadas» (¿alias? ¿plural?
   ¿la línea no está en el catálogo?) y arregla lo que sea del instrumento.
3. Re-puntúa con el estricto contra la línea base del lote 38: recall por clase antes/después y **FP nuevos sobre los 9 `ok`**.
   Objetivo, no promesa: `coccion_faltante` ≥ 6/8 sin FP nuevos en los `ok`; si no llega, el informe dice a cuánto llegó y por qué.
4. Test `tests/test_p1_plan_lote_39.py`: V7f dispara en un plato mínimo (arroz crudo + «incorporar»), no dispara con «hierve el
   arroz 15 min», no dispara sobre un listo-para-comer; RUBRICA trae V7f; knob apagado ⇒ 0 hallazgos; docs + marker ≥ 39.

### Criterio de terminado

Tabla recall/FP por clase antes/después en el doc del área (`culinary_coherence.md` §C5, con fecha); ningún detector cambia de `warn`
a bloqueo; `culinary_golden_set.json` no se edita a mano (sólo el refresco del lote 38).

---

## Lote 40 · Calibración del juez (la parte medible de C5/C6)

### Medido

Contra la verdad humana sin el filtro de alimento: juez precisión **11,8 %** (4 tp / 30 fp / 5 fn). Sus 36 hallazgos del 09-06:
`paso_incoherente` 12 (3 tp), `nombre_no_corresponde` 8 (1 tp), `tecnica_impropia` 7, `combo_absurdo` 5, `slot_inapropiado` 4 (0 tp
los tres; al menos 2 de ellos son `seco_sin_coccion` del dueño bajo otro nombre). `MEALFIT_CULINARY_JUDGE_GUARD` nace `off` en
producción (C6); promoverlo es del dueño **cuando C1 haga interpretable el veredicto** — este lote es eso.

### Qué hacer

1. Con las columnas refrescadas (lote 38, `--con-juez`), tabla por código del juez: n, tp, fp, y de los fp cuántos son «mismo defecto,
   otro código» (lee la evidencia del dueño del mismo caso: si nombra el mismo alimento y el mismo problema, es un **error de código,
   no de juicio**). Publica las dos precisiones: la estricta y la «por sustancia».
2. Códigos con precisión estricta < 25 % y n ≥ 4 pasan a **observación**: el post-proceso del juez los marca `[dudosa]` (ya existe la
   marca del lote 28; `culinary_golden_score.py` no los cuenta salvo `--con-dudosas`). Lista en el knob
   `MEALFIT_CULINARY_JUDGE_OBSERVACION_CODES` (default: los que salgan de la medición, escritos en el doc con su cifra).
3. Mapa de códigos del juez a clases de la RUBRICA para la adjudicación «por sustancia» (`slot_inapropiado`/`tecnica_impropia` sobre un
   crudo ⇒ `seco_sin_coccion`), SOLO en el marcador, nunca en producción: el instrumento puede ser generoso; el juez no cambia de voz.
4. **Recomendación para C6**, escrita, con las cifras: con qué precisión estricta y por sustancia se quedaría `warn`, qué códigos
   quedarían en observación, y qué falta (2.º anotador). No se flipa nada.
5. Test `tests/test_p1_plan_lote_40.py`: el post-proceso marca `[dudosa]` los códigos del knob y ninguno más; el mapa por sustancia
   vive en el script, no en `culinary_coherence`; docs + marker ≥ 40.

### Criterio de terminado

Doc `culinary_coherence.md` §C6 con la tabla por código y la recomendación firmada con fecha; knob registrado; el juez de producción
emite los mismos hallazgos que antes, sólo cambia la marca de los códigos en observación.

---

## Lote 41 · La tormenta de reintentos del catálogo con la base caída

### Medido (2026-09-13, CI sin base)

`build_blueprint` hace **32.564** llamadas a `shopping_calculator.get_master_ingredients` y, sin pool, CADA una entra en la rama
`else` (`shopping_calculator.py:645-648`), registra `logging.error("No connection_pool available to fetch master_ingredients")` y
devuelve `[]` sin sellar `_master_cache_ts`: **28.611** líneas iguales en un job (103789056311). `catalog_capability.catalog_capability`
(`catalog_capability.py:84-131`) sólo cachea snapshots NO vacíos: con `rows == []` devuelve `None` sin cachear y vuelve a llamar en la
siguiente; su `_avisar_una_vez` deduplica su propio aviso, no la llamada. La cola de la CI que esto alimentaba ya no existe (Sentry
sin DSN no serializa, `P1-PLAN-LOTE-37`), pero el patrón sigue vivo para producción: **un arranque en frío con Neon caído convierte
cada blueprint en decenas de miles de intentos de conexión y líneas de log**.

### Qué hacer

1. **Caché negativa corta** en `get_master_ingredients`: en las ramas «sin pool» y «excepción», sella `_master_cache_neg_until = now +
   MEALFIT_CATALOG_NEGATIVE_CACHE_S` (knob, default **30** s, clamp [1, 300]) y registra el error UNA vez por ventana (las siguientes,
   `debug` con el conteo). Mientras dure la ventana, devuelve `_master_cache or []` sin tocar el pool. **Nunca** selles
   `_master_cache_ts` en esas ramas: eso es exactamente lo que `P1-CATALOG-INDEX-NO-STICKY` (comentario en 624-632) prohíbe — servir
   5 minutos de vacío como si fuera el catálogo. Si el pool aparece antes de que venza la ventana, se ignora la ventana (comprueba
   `connection_pool` primero: los tests parchean `db_core.connection_pool` en caliente).
2. El `reset` de cachés de `shopping_calculator.py:434-436` limpia también el sello negativo.
3. `catalog_capability`: cachea el `None` por país con el mismo TTL corto (`_CACHE_NEG[cc] = (None, until)`), invalidado por el mismo
   reset. La semántica no cambia: `None` sigue significando «capacidad desconocida, no cero».
4. Medir antes/después con un script del scratchpad: `db_core.connection_pool = None`, `horizon.build_blueprint(...)` de un perfil del
   landing, contar llamadas al pool (parchea `execute_sql_query`) y líneas de log. Objetivo: de 32.564 intentos y 28.611 líneas a
   ≤ 2 intentos y ≤ 2 líneas por ventana de 30 s.
5. Test `tests/test_p1_plan_lote_41.py`: sin pool, 1.000 llamadas ⇒ 1 error de log y 0 accesos al pool dentro de la ventana; al vencer,
   reintenta; con pool que aparece a mitad de ventana, lee la tabla en la siguiente llamada; el sello negativo NUNCA escribe
   `_master_cache_ts`; el reset limpia el sello; knob registrado y documentado; marker ≥ 41.

### Criterio de terminado

`knobs_reference.md` con el knob y las dos cifras; `shopping_calculator.py` no supera 14.400 líneas; `catalog_capability.py` sin
cambiar la firma pública; la tarea propuesta del 13-sep («Stop the catalog retry storm when the DB is down») queda cerrada citando el
lote.

---

## Lote 42 · Worktrees viejos (E1, la parte operativa)

`git -C backend worktree list` (2026-09-14, tras retirar el scratch `C:/tmp/ci_tail/backend`):

| Worktree | HEAD | Rama |
|---|---|---|
| `C:/tmp/arq25/backend` | b4684a02 | `glm-migration` |
| `C:/tmp/mf-audit/backend` | ca41f624 | (detached) |
| `C:/tmp/mf-be-i18n` | 054cb655 | `i18n-display` |
| `C:/tmp/mf-wt/backend` | af00b5c7 | `paises-ola-2` |
| `C:/tmp/mfi18n/backend` | de1e90fd | `i18n-v3-p0` |
| `C:/tmp/neon_migration/wt_pre_vps` | 6b0ee297 | (detached) |
| `C:/tmp/neon_migration/wt_prerewrite` | 7f9f54a0 | (detached) |
| `…/MealfitRD.IA/.claude/worktrees/vibrant-bhabha-f98d2f/backend` | 866ac4db | `claude/p1-aviso-capado-lee-el-envase` |
| `…/MealfitRD/Software/mf-wt-backend` | a7219e2c | `i18n-lotes` |

Regla, sin excepciones: se retira un worktree **sólo si** `git -C <wt> status --porcelain` está vacío **y** su commit es ancestro de
`main` (`git merge-base --is-ancestor <sha> main`) o su rama está en `git branch --merged main`. `git worktree remove` (sin `--force`),
luego `git worktree prune`. **Ninguna rama se borra**: eso es G4, del dueño. Los que no cumplan la regla van en una tabla al informe con
el motivo (sucio / no mergeado / rama viva). Sin marker ni deploy: no cambia un byte del repo.

---

## Lote 43 · Re-mediciones con fecha (B6 · E5 · D7 · B9)

El lote 36 dejó tres instrumentos de solo lectura y un veredicto «sin muestra, con fecha»: `scripts/measure_variety_windows.py` (B6),
`scripts/measure_canonical_shadow.py --offline` (E5, hoy 5 de 30 planes: no concluyente) y `scripts/measure_undersupply_volume.py`
(D7: 3 `magnitude_undersupply` en 45 días, todos de la ráfaga del canario 09-04/05; **volver el 2026-10-10**). B9 (embudo del wizard) exige ≥ 2 semanas de
`step_done` reales desde el lanzamiento. En su fecha: correr los tres tal cual, pegar las cifras en las filas B6/E5/D7 del plan con la
fecha, y **no decidir nada**: los flips (`MEALFIT_DETERMINISTIC_DAY_SAME_DAY_VARIETY`, fase B de E5, `MEALFIT_GUARD_UNDERSUPPLY_SEVERE`)
son del dueño. Antes de la fecha, no se corre: una medición repetida sobre la misma muestra vacía no añade información.

---

## Lote 44 · Aplicar las decisiones del dueño (delegadas el 14-sep)

Registro y razones: [`decisiones_dueno_2026_09_14.md`](decisiones_dueno_2026_09_14.md). La hoja (documento `angelo`) tiene el
texto íntegro de las recetas. Cada punto se mide antes y después con el instrumento que ya existe; nada de esto se decide
de nuevo aquí.

### Qué hacer

1. **Recetas** (C4): copia los pasos de la hoja a `data/registry/recipe_library_do_v1.json` (`por_id[<tpl>].pasos`) para
   `tpl_14c76a1c346e`, `tpl_a7799418aa9c`, `tpl_fc758e30f5e7`, con `procedencia` «dueño 2026-09-14 (hoja), redacción del agente
   por delegación». Corre `python scripts/asignar_uso_pasos.py --write --verificar` (lote 25): las tres deben quedar cerradas en
   `recipe_usage_do_v1.json`. Si la biblioteca o el registry llevan hash anclado en tests (lo llevan: F6, «5 bibliotecas
   re-ancladas con lo MEDIDO»), recompila con el script que nombra `docs/dish_registry_f6.md` y re-ancla con el valor medido; DO
   sigue 193/193 `ok`. No toques ninguna otra receta.
2. **Claras en botella a partir de 4** (C3): localiza dónde la compra colapsa claras y yemas en cartones de `Huevo` (busca
   `MAX_EGG_WHITES_PER_MEAL`, `_cap_daily_whole_eggs` y la decisión del 11-may en `knobs_reference.md`). Knob
   `MEALFIT_EGG_WHITE_BOTTLE_MIN_PER_MEAL` (default **4**, clamp [1, 12]; `0` = nunca botella): si una comida pide ≥ N claras, la
   línea de compra resuelve a la fila de claras pasteurizadas del supermercado («Clara de huevo · Don Papito · botella 400 g»;
   comprueba que la fila maestra y su mapeo existen, y que el `fdc_id` de clara ya es propio — lo es desde el lote 24); por
   debajo, cartones como hoy. Mide primero en el corpus fijo cuántas comidas cruzan el umbral (14 de 64 llevan huevo; 6
   salieron del tope de enteros) y anótalo en `culinary_coherence.md` §C3.
3. **Lista canónica, fase B, cohorte = canario del dueño** (E5): knob `MEALFIT_CANONICAL_SHOPPING_USERS` (lista de ids; default:
   **el mismo valor que `MEALFIT_DETERMINISTIC_DAY_USERS`**, para que la cohorte nazca igual al canario sin tocar el `.env` del
   VPS). Para esos usuarios la lista entregada es la canónica, con los topes del agregador heredados tal cual; la sombra y su
   métrica (`canonical_shopping_shadow`) siguen para TODOS. Diseño y trampas ya pagadas: `docs/arq30_e5_e7_diseno_canario.md`
   fase B. Kill switch: lista vacía.
4. **Pesos del día determinista solo en el canario** (B7): knob `MEALFIT_DETERMINISTIC_DAY_W_CARB_SURPLUS_CANARY` (default **2.0**)
   que se aplica en lugar del global (`_W_CARB_SURPLUS`, sigue 1.0) cuando el usuario está en `MEALFIT_DETERMINISTIC_DAY_USERS`;
   `_W_FAT_DEFICIT` queda 1.0. Mide antes/después con `scripts/measure_deterministic_day_macros.py` para el perfil del canario y
   pega las cifras en `docs/deterministic_day.md` (hoy: carbohidrato +18 % / grasa −18 %).
5. **PayPal `block`** (G1): `MEALFIT_BILLING_VERIFY_AMOUNT` default `block` en `routers/billing.py` (~línea 635): sin cupón activo,
   un importe por debajo del precio del plan bloquea la activación y persiste alerta (si el `alert_key` es nuevo, fila en
   `docs/system_alerts_resolution_table.md`: `test_p2_audit_4` exige paridad). Reproduce el escenario del
   `paypal_audit_2026_08_22.md` §override en test; `I-Billing-1/2/3` intactas.
6. **Sin código**: D8 se cierra en el plan («no construir»); F9 anota al nutricionista; V7a y equipo, sin cambios. La calibración
   del juez (sí, tope $1) es DESPUÉS del lote 38 y nunca sobre el instrumento roto.

### Criterio de terminado

Tests `tests/test_p1_plan_lote_44.py` (recetas cerradas en `recipe_usage`, umbral de claras con 3 y 4, cohorte B por knob con
kill switch, peso del canario no toca el global, `block` sin cupón bloquea y con cupón válido no); knobs registrados y en
`knobs_reference.md`; marker ≥ 44; gate; deploy; `/health/version` sin drift. Lo que NO se hace: mover topes de la canónica,
promover pesos al global, calibrar al juez.

---

## Lote 45 · La prueba RD del dueño (HECHO · `P1-PLAN-LOTE-45` · 2026-09-14)

No estaba en el plan: salió de revisar el plan `40535829` que el dueño generó con su perfil dominicano. Cerrado en otra
sesión; se anota aquí para que nadie lo rehaga. Qué cambió: el reparador del contrato rotula el pilar en su sitio
(`recipe_order`) en vez de extraer oraciones —149 de 193 recetas de biblioteca salían desordenadas— y `repair_stage_diff`
cuenta el desorden; el día determinista toma la familia del blueprint y el tiempo de cocina del formulario; el pool del
planificador ya no lo rechaza; el reintento siembra la memoria con los días reciclados. Pendiente y medido: los
candidatos fijados por franja se leen detrás de `MEALFIT_DETERMINISTIC_DAY_PINNED_SLOT_ALIAS` (apagado), y encenderlo pide
antes un CandidateSet más ancho que sepa del tiempo. Detalle: `deterministic_day.md` y `culinary_coherence.md`.

## Lote 46 · La segunda prueba RD del dueño (HECHO · `P1-PLAN-LOTE-46` · 2026-09-14)

Tras el lote 45 el dueño repitió la prueba (plan `63eedc6b`). Cerrado en otra sesión: el día determinista prefiere lo no
servido en los últimos 3 planes y respeta el tope de repetición del bloque; la familia del blueprint filtra sólo la comida
principal y la puerta de proteína del día evita repetirla; el revisor no reintenta por repetidos ni por proteína repetida en
días deterministas; el ingrediente que da nombre al plato no lo recortan los re-trims y, si falta, vuelve al 25 %
(`identidad_plato`); el cerrador no manda hervir el queso y el autofix no reescribe recetas congeladas. Abierto: «si no
cabe, se cambia el plato» (elegir otra plantilla cuando la identidad no cabe en los macros) no está hecho: hoy la identidad
vuelve y el truth-up la cuenta.

## Lote 47 · La tercera prueba RD del dueño (HECHO · `P1-PLAN-LOTE-47` · 2026-09-14)

Tras el lote 46 el plan (`d8b10b05`) salió aprobado al primer intento y sin repetir los planes anteriores, pero la
autocrítica corrigió con el LLM los días 1 y 2 y la regeneración quirúrgica el 3: el corrector pegó encima días sin
procedencia y los 6 platos de biblioteca que conservó por el nombre perdieron las protecciones de los lotes 45 y 46.
Cerrado en otra sesión: re-elegir con el armador en vez de reescribir (`reeleccion_dia`), procedencia restaurada en lo que
el corrector deja igual, las reglas fijas de la autocrítica en la elección del día determinista y tres parches de pasos
(`pasos_sustitucion`). Abierto: «Nada» de tiempo de cocina en almuerzos y cenas (la biblioteca no tiene con qué: los
minutos medios del día determinista subieron de 20,9 a 25,9 al cumplir las reglas).

## Lote 48 · La cuarta prueba RD del dueño (HECHO · `P1-PLAN-LOTE-48` · 2026-09-14)

Tras el lote 47 el plan (`358a2cdf`) llegó con 11 de 12 comidas de biblioteca y aprobado al primer intento, pero los
cerradores de macros les colgaban cosas que la receta no haría (arroz junto al mofongo, queso licuado en un jugo, «queso»
a secas que la lista compraba como queso blanco), el cambio del arenque borraba la frase del locrio con el arroz dentro,
la chinola llegó a 335 g, y un día que el barrido de la re-elección ya había dejado limpio fue igual al LLM. Cerrado en
otra sesión: `cierres_con_receta` (base propia, jugo, queso con nombre), el desalado como cláusula, el techo de la pulpa,
el piso de proteína como preferencia del armador (peso 0,4; medido: con 1,5 repetía un plato 4 veces en 7 días) y la cola
re-medida. Abierto: en la réplica del run la autocrítica aún saltaría por «yuca en 2 días» (y «pollo en 3» en el segundo
bloque) —la recoge la re-elección del 47, sin LLM— y «Nada» de tiempo de cocina sigue sin alternativa en la biblioteca
(29 min medios).

## Lote 49 · La quinta prueba RD del dueño (HECHO · `P1-PLAN-LOTE-49` · 2026-09-14)

El plan `a059d7bb` fue el mejor de la serie (144 s, 3/3 días deterministas, 12/12 recetas de biblioteca, juez 0), con
defectos a la vista: el ingrediente que da nombre en migajas (guacamole con 5 g de aguacate), el cerrador repitiendo el
huevo del día y pegando 10 g de arenque, su paso dentro del primer paso con fuego, «3 claras» en la lista y huevo entero
en la compra, 6/12 platos de un plan anterior (la fila vacía del plan en curso ocupaba un hueco) y 7/12 comidas sobre el
tiempo de «Nada». Cerrado en otra sesión. Abierto: (1) el lector de la lista del contrato lee mal los decimales con punto
(«57.2 g» → 2 g); (2) «Nada» no tiene platos: la biblioteca DO tiene 2 de 63 almuerzos y 1 de 56 cenas en 10 min — la
decisión (redactar almuerzos y cenas rápidos) es del dueño; (3) decir en el panel «solicitaste / aplicamos / por qué» el
tiempo que no se pudo cumplir necesita un código de motivo nuevo en el frontend (5 idiomas).

## Para el dueño (no lo cierra código)

Las cuatro tareas de GitHub de la sección 4 de la [Hoja del dueño](https://claude.ai/code/artifact/fa74e5bb-85be-446c-9598-3fda3960c503)
(`SIBLING_REPO_TOKEN`, los 4 secretos del benchmark nocturno, las ramas superadas, el 2.º anotador), la fecha de la revisión de
Juan Carlos Brito (F9), la re-firma curatorial si cambia algo del registry, la tabla del ICBF en Excel (F7: Achiote, Chontaduro,
Champús), el recorrido humano de i18n en la app nativa (F3), `PYTHONHASHSEED` en la unidad systemd (B4), la decisión sobre E3, y el
flip de C6 cuando el lote 40 le ponga cifras delante. Las diez decisiones del 14-sep las eligió el agente por delegación
(`decisiones_dueno_2026_09_14.md`): si alguna no le convence, se revierte cambiando su knob.

---

## Informe final al dueño (plantilla, por lote)

Marker y commit · qué se midió ANTES y qué salió · qué se cambió (ficheros) · cifras después, con la misma vara · gate (A/B/C con
conteos) · CI (run, minutos por pata) · deploy y `/health/version` · **lo que NO se hizo y por qué** · lo que queda para el dueño, con
la decisión exacta que se le pide. Sin adjetivos: si una cifra no cambió, se dice que no cambió.
