# Coherencia culinaria determinista — `P1-CULINARY-CONTRACT`

> Doc canónica de F1 del diseño [`2026-07-31-culinary-coherence-design.md`](../../docs/superpowers/specs/2026-07-31-culinary-coherence-design.md) (secciones 4-4c). CLAUDE.md tiene el 1-line + link; el detalle vive aquí. Golden set (F0): [`P1-CULINARY-GOLDEN`](../tests/test_p1_culinary_golden.py) + fixtures [`backend/tests/fixtures/culinary_golden/`](../tests/fixtures/culinary_golden/).

Cierra la causa raíz "el conocimiento culinario vive en ~6 tuplas hardcodeadas
reactivas" (`_NO_COOK_SAFE_PROTEIN_HINT`, `_PRECOOKED_PROTEIN_HINT`,
`_LEGUME_PROTEIN_HINT`, `_COOKED_GRAIN_REF_KCAL`, `_COOKING_VERB_RE`) con
metadata SSOT en `master_ingredients` + un validador determinista puro. Esta
doc documenta lo **construido** (Tasks 1-8 de este SDD), con sus decisiones
reales — no lo planeado por el spec original, que a veces difiere del código
final (ver "Divergencias del spec" al final de cada sección).

---

## Las 3 capas y su estado

| Capa | Qué hace | Estado |
|---|---|---|
| **Capa 1 — scan determinista** | Metadata `prep_methods`/`ready_to_eat` en `master_ingredients` + `culinary_contract_scan()` (V1 verbo↔alimento, V2 estado imposible, V3 huérfanos) en 3 superficies. | **`warn`** (F1, esta doc). Escalada a `block` es F2 (`P1-CULINARY-CONTRACT-BLOCK`), no implementada. |
| **Capa 2 — juez LLM** | `run_culinary_judge()` (DeepSeek flash) ve recetas completas (el reviewer médico no las ve) y detecta clases abiertas (combos raros, técnica impropia, nombre no corresponde) vía `CulinaryJudgeReport`, integrado en `review_plan_node` con history propio (`_culinary_judge_history`). | **Implementada, nace `off`** (`P1-CULINARY-JUDGE`, Tasks 11-12). Calibrada 2026-08-01 (ver sección abajo) — criterios cumplidos, autoriza la escalada OFF→`warn`; activarla en prod es decisión de owner (F4, Task 14). |
| **Capa 3 — calibración medida** | Script `scripts/calibrate_culinary_judge.py`: recall/FP por clase y por capa contra el golden set, con llamadas LLM reales (no CI). | **Hecha** (Task 13, 2026-08-01). Resultado: capa1 recall 100% + 0 FP (contrato F1 re-confirmado); juez recall 89% + FP 0% (criterios spec §6 cumplidos tras 1 iteración de la rúbrica). |

Rollout completo (spec §7): F0 golden set (hecho) → **F1 capa 1 en warn (hecho)** → F2 capa 1 en block (pendiente) → **F3 juez OFF calibrado (hecho, esta doc, 2026-08-01 — criterios cumplidos, autoriza OFF→warn)** → F4 juez warn→block medido (pendiente, requiere ≥1 semana de warn limpio en prod, Task 14).

---

## Tabla de superficies (F1)

Espejo de [`coherence_surfaces_table.md`](coherence_surfaces_table.md) (mismo patrón: tabla "negativa" — qué SÍ bloquea vs qué solo mide).

| # | Superficie | Archivo | Modo | Qué garantiza |
|---|---|---|---|---|
| 1 | `review_plan_node` | [`graph_orchestrator.py:38280+`](../graph_orchestrator.py) | gate según `MEALFIT_CULINARY_CONTRACT_GUARD` (default `warn`) | Corre DESPUÉS del AUTO-PATCH de huérfanos y AUTO-PATCH-FORWARD (dueño único: reparar → medir). Escribe `plan["_culinary_contract_violations"]` + `plan["_culinary_contract_coverage"]` siempre. En `warn` solo loggea. En `block` (no es el default) cada violación se traduce a `issues.append(...)` + `_severity_max`, y por diseño F1 cae a **retry completo** — "incoherencia culinaria" no está en `_SURGICAL_REJECT_SAFE_PREFIXES` ni en `_SURGICAL_REJECT_REJUDGED_PREFIXES`, así que no hay ruta quirúrgica per-día para este gate todavía (ver "Divergencias" abajo). |
| 2 | `finalize_plan_data_coherence` | [`graph_orchestrator.py:25000`](../graph_orchestrator.py) (`_fix_refill_step_verb`, def en `:28138`) | reparación | Cierra la paridad assemble↔finalize: `_fix_refill_step_verb` (repara pasos tipo "🍚 Cuece el Casabe" → "Sirve el Casabe") corría solo en `assemble_plan_node`; ahora también corre en el loop tardío `P2-MISE-COOK-SPLIT` de finalize, en el mismo orden relativo que assemble (`_align_closer_note_food_names` → `_split_cooking_from_mise` → `_fix_refill_step_verb`, al final del trío — el defecto lo introduce un renombrado posterior al productor del paso). Idempotente + fail-safe: re-correrlo donde assemble ya lo aplicó es no-op. |
| 3 | Path degradado (`_build_filtered_edge_recipe_day`, `cron_tasks.py`) | [`cron_tasks.py:24541+`](../cron_tasks.py) | scan + reparación (única capa posible ahí) | Este path NUNCA pasa por `assemble_plan_node`/`review_plan_node` — no hay LLM por construcción. Dos mejoras: (a) el verbo del paso de Desayuno se deriva de `prep_methods[0]` real del alimento en vez de un placeholder genérico fijo ("según método tradicional"); (b) el día ENSAMBLADO pasa por `culinary_contract_scan` (solo V1/V2 — V3 se tolera, ver abajo) y `_degrade_offending_steps` degrada el paso ofensor a `"Sirve el {food}."`, acotado por `meal` exacto y con el matcher canónico `find_catalog_foods` (no substring plano). Fail-open total: si el scan revienta, el día sale tal cual — el backstop de seguridad (`P0-DEGRADED-SAFETY-SCAN`) es una capa aparte y anterior. |

**V3 (huérfanos) NO corre en la superficie 3**: los Edge Recipes son 3 pasos fijos (Mise en place / El Toque de Fuego / Montaje) que nunca listan cada ingrediente paso a paso — aplicar V3 ahí degradaría pasos que ya están bien. V1/V2 sí, porque verbo y estado se pueden falsar sin necesitar que el paso enumere cada ingrediente.

**El bloque de degradación de la superficie 3 es un no-op estructural hoy** (documentado, no un bug): el verbo del Desayuno se deriva del `prep_methods[0]` del MISMO alimento que luego valida el scan, así que nunca puede violar V1 por construcción; los pasos de Almuerzo/Cena no mencionan el nombre del alimento en el texto, así que `find_catalog_foods` no encuentra a quién acusar. Es exactamente lo que el spec pide ("última palabra por si algo se coló") — un backstop de defensa-en-profundidad para ediciones FUTURAS de las plantillas, verificado con una violación inyectada a mano (ver `test_degrade_offending_steps_matcher_canonico_no_substring`), no algo que dispare hoy con las plantillas actuales.

---

## Tabla de knobs

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CULINARY_CONTRACT_GUARD` | `warn` | `off`/`warn`/`block`. Clamp a `warn` si el valor no es uno de los 3. Registrado vía `_env_str` → `_KNOBS_REGISTRY` ([`graph_orchestrator.py:6112`](../graph_orchestrator.py)). Escalada a `block` es F2 (`P1-CULINARY-CONTRACT-BLOCK`), pendiente. |
| `MEALFIT_CULINARY_JUDGE_GUARD` | **`off`** | `off`/`warn`/`block`. Clamp fail-safe a `off` (no a `warn`) si el valor no es uno de los 3 — a diferencia del contract-guard, esta es una llamada LLM completa por plan, así que el único default seguro ante un valor raro es apagado. Calibrado 2026-08-01 (criterios cumplidos); activar `warn` en prod es decisión de owner (Task 14). |
| `MEALFIT_CULINARY_JUDGE_MODEL` | `_FLASH_MODEL_NAME` (DeepSeek flash) | Directiva del owner (P1-FLASH-PRIMARY) — nunca pro sin medición. Override para A/B de modelo/costo. |
| `MEALFIT_CULINARY_JUDGE_THINKING` | `False` | Activa `extra_body.thinking` (DeepSeek-only, ignorado en modelos OpenAI). No probado en la calibración 2026-08-01 (baseline es sin thinking). |
| `MEALFIT_CULINARY_JUDGE_TIMEOUT_S` | `45` | Clamp `[10, 120]`. Timeout de la llamada al juez; `asyncio.wait_for` externo da `+5s` de margen. |

Los 4 knobs de la Capa 2 se leen a **import-time** de `graph_orchestrator.py` (constantes módulo-level, no releídas por llamada) — un script que necesite forzar `warn` para calibrar debe escribir `os.environ` ANTES del primer `import graph_orchestrator` (ver docstring de `scripts/calibrate_culinary_judge.py`, "TRAMPA Nº1").

---

## Vocabulario `prep_methods` (SSOT)

10 métodos canónicos, el mismo set en la columna Postgres, el sanity `DO $$` de la migración, y `culinary_coherence.PREP_VOCAB`:

```
hervir | plancha | freir | hornear | guisar | saltear | licuar | tostar | crudo | ninguno
```

Mapeo verbo-de-receta → método (`VERB_TO_METHOD` en [`culinary_coherence.py`](../culinary_coherence.py)), con 2 fusiones deliberadas descubiertas por el golden set real (no por los tests unitarios con catálogo sintético — el catálogo sintético no tiene alimentos con calificador recortable ni verbos ambiguos):

- **`sofr[ií]\w*`/`dora\w*` → `saltear`** (NO `freir`/`tostar`). Sofreír cebolla/ají es la base de casi toda receta dominicana y la metadata de Vegetales lleva `saltear` pero no `freir`; "dora" (sellar carne/pollo en sartén, primer paso de casi todo guiso) NO es tostar pan/casabe — `tostar` no está en `prep_methods` de proteínas frescas. Ambas fusionadas en la MISMA clave que `saltea\w*` (no en claves separadas) porque dos claves resolviendo al mismo método duplicaban la violación V1 vía `dict.fromkeys` — sin el dedup, "Sofríe y saltea el X" producía 2 violaciones idénticas.

**Política NULL = fail-open, por check**: si un alimento no tiene `prep_methods`/`ready_to_eat` (columna `NULL`, DEFAULT de la migración), el scan se salta ESE check para ESE alimento — nunca inventa, nunca asume `false`. `scan_coverage()` mide la fracción de alimentos del plan con metadata (telemetría del rollout warn→block).

**Cobertura actual** (migración base, antes del backfill `leche%hervir`): `prep_methods` 148/204 filas de `master_ingredients` (~72.5%), `ready_to_eat` 99/204 (~48.5%, menor porque Vegetales/Víveres solo setean `prep_methods` por diseño del backfill de categoría). El backfill `leche%hervir` solo AÑADE un método a filas ya no-NULL (no cambia el conteo de cobertura). El resto queda `NULL` — fail-open, no un gap de esta task; un audit de huecos fuera del golden set queda pendiente para una task futura.

---

## Regla "reparar → medir → juzgar" y dueños únicos

Orden fijo (spec §9, "Oscilación reparador↔gate↔juez"): los reparadores deterministas mutan el plan PRIMERO, el scan mide el residuo DESPUÉS, y el juez (F3) solo juzga, nunca muta. Dos dueños concretos hoy:

- **AUTO-PATCH** (`review_plan_node`, bloque `"huérfanos eliminados"`) **repara** — ELIMINA ingredientes huérfanos de la lista antes de que corra el scan.
- **V3** (`_v3_huerfanos` en `culinary_coherence.py`) **mide el residuo** — corre DESPUÉS del AUTO-PATCH (y de AUTO-PATCH-FORWARD), nunca antes. Nunca los dos deciden sobre el mismo caso: si el AUTO-PATCH ya borró el huérfano, V3 no lo ve.

Mismo patrón en la superficie 2: `_fix_refill_step_verb` repara ANTES de que cualquier lint/contrato posterior mida — anclado por el orden real del trío `_align_closer_note_food_names → _split_cooking_from_mise → _fix_refill_step_verb` (ambas cadenas, assemble y finalize).

---

## Migraciones

Ambas idempotentes (`IF NOT EXISTS` / `array_append` con guard `NOT (... = ANY(...))`), copiadas byte-idénticas a `migrations/` (root) y `backend/migrations/` (P3-MIGRATIONS-SSOT), aplicadas a Neon y verificadas no-op en 2ª corrida:

1. [`p1_culinary_metadata_master_ingredients_2026_07_31.sql`](../migrations/p1_culinary_metadata_master_ingredients_2026_07_31.sql) — añade `prep_methods text[]` + `ready_to_eat boolean` (ambos `DEFAULT NULL`) a `master_ingredients`. 3 backfills en cascada (defaults por `category` → tuplas hardcodeadas históricas → ~30 overrides explícitos de casos delicados como casabe/embutidos/enlatados/legumbres), sanity `DO $$` que revienta si algún `prep_methods` sale del vocabulario canónico.
2. [`p1_culinary_metadata_leche_hervir_2026_07_31.sql`](../migrations/p1_culinary_metadata_leche_hervir_2026_07_31.sql) — hueco de metadata cazado por el golden set (no por revisión manual): el backfill por categoría dejó `Lácteos` sin `hervir`, y "Hierve la leche" (avena, café con leche) es cocina dominicana básica → falso positivo V1 real contra 4/5 planes buenos de `golden_XX_bueno`. Alcance `leche%` (8 filas), no solo la fila `Leche` que el examen pisó — cierra la misma mina en el resto de la categoría.

---

## Golden set (F0, cimiento de este scan)

10 fixtures estáticos commiteados en [`backend/tests/fixtures/culinary_golden/`](../tests/fixtures/culinary_golden/): `golden_{01..05}_bueno.json` (creatividad legítima, miden falsos positivos) + `golden_{01..05}_mutado.json` (4-6 defectos inyectados y etiquetados por `golden_manifest.json`, miden falsos negativos), cubriendo los 4 slots y ≥1 vegetariano, construidos desde la DB dominicana real. Generador one-shot `scripts/build_culinary_golden_set.py` — estáticos a propósito, el ground truth no se reescribe en silencio si cambia la DB.

**Criterios de F1** (spec §6, ya en CI, deterministas — sin flakiness):
- **0 falsos positivos**: `culinary_contract_scan` sobre los 5 `golden_XX_bueno` contra el catálogo REAL de Neon → 0 violaciones (`test_capa1_cero_fp_sobre_los_buenos`).
- **100% de las clases `capa1:*`**: cada defecto inyectado en los 5 `golden_XX_mutado` con `expected_by` empezando en `capa1:` (verbo_imposible, estado_imposible, ingrediente_huerfano, tecnica_impropia) queda atrapado por el check correcto en el día correcto (`test_capa1_atrapa_100pct_de_sus_clases`). Si un test golden falla, el fix va al scan o a la metadata — **jamás relajar el fixture**.

**Cómo re-calibrar** si el catálogo cambia y el examen vuelve a rojo: correr `pytest tests/test_p1_culinary_golden.py -q` contra Neon real (`NEON_DATABASE_URL` seteada, pool abierto — los tests golden hacen `pytest.skip` limpio sin DB, así que CI de GitHub no los ejecuta hoy). Un FP nuevo casi siempre es o (a) un hueco de metadata real → migración SSOT nueva con alcance amplio (patrón `leche%hervir`, no solo la fila puntual), o (b) un verbo ambiguo mal clasificado en `VERB_TO_METHOD` → mover de grupo con el razonamiento documentado inline (patrón `dora`/`sofr`). Nunca silenciar el fixture para que pase.

3 falsos positivos reales encontrados por el examen contra Neon (no por los tests unitarios sintéticos): `Arroz blanco` mencionado por su forma genérica "el arroz" (cerrado con `_mencionado_por_prefijo`, con guard de ambigüedad contra prefijos compartidos tipo "Ají morrón"/"Ají cubanela"), `dora` mal clasificado bajo `tostar` (cerrado moviéndolo a `saltear`), y `Leche` sin `hervir` en su metadata (cerrado con la migración `leche%hervir`, no en el scan — es un hueco de datos, no de lógica).

---

## Juez LLM (Capa 2, F3): rúbrica y calibración

`run_culinary_judge(plan)` ([`graph_orchestrator.py`](../graph_orchestrator.py)) hace UNA llamada batched (no por día) a `CULINARY_JUDGE_MODEL` con `with_structured_output(CulinaryJudgeReport)`. Fail-open total: knob `off`, timeout, o cualquier excepción del LLM/parseo → `None`, nunca bloquea el plan por su cuenta. Ve la receta completa por plato (`recipe`, pasos) — el único ojo LLM del pipeline que la ve; el reviewer médico solo recibe nombre+ingredientes.

`CulinaryViolation.tipo` acepta exactamente 5 valores canónicos: `combo_absurdo`, `tecnica_impropia`, `paso_incoherente`, `slot_inapropiado`, `nombre_no_corresponde`. La rúbrica (`_CULINARY_JUDGE_RUBRIC`, construida UNA vez a import-time para cache hits de DeepSeek sobre el prefix estable) combina: hasta 10 nombres de ejemplo por slot desde `data/dish_templates.json`, la REGLA DURA de horario (arroz/locrio/moro/pasta nunca en desayuno/cena; sopones solo en almuerzo; postre como plato principal), la GUÍA POSITIVA por horario de `constants.SLOT_POSITIVE_HINT`, y las definiciones de los 5 tipos con ejemplos.

**Iteración de rúbrica (2026-08-01, 1 ronda de 3 permitidas — spec §6):** la corrida baseline inicial con la rúbrica original de Task 11 dio juez recall 78% + FP 16.7% (**FALLA** ambos criterios). Causa raíz de los 6 FPs: la GUÍA POSITIVA de `cena` dice "evita... los guisos pesados" — el juez la trataba como regla dura y marcaba `slot_inapropiado` sobre guisos de proteína legítimos como cena (pollo/carne/pescado guisado), que SÍ son cena dominicana real y ninguno de los 5 `golden_XX_bueno` los etiqueta como defecto. Causa raíz de los 2 misses de recall: (a) el swap `golden_02_mutado` renombra "Moro de habichuelas negras" → "Moro de guandules" **sin cambiar la categoría del plato** (sigue siendo arroz+legumbre, solo cambia la legumbre) — más sutil que los otros 3 swaps de `nombre_no_corresponde` del golden set, que sí cambian de categoría de plato; (b) el defecto de `tecnica_impropia` en `golden_03_mutado` (yogurt sobre la plancha) lo detectaba pero lo etiquetaba `paso_incoherente` — ambigüedad de frontera entre "técnica mal aplicada en un paso" y "dos pasos que se contradicen".

Fix (`_build_culinary_judge_rubric()`, mismo archivo): (1) la GUÍA POSITIVA se rotuló explícitamente "orientativa, NO una regla dura" + un párrafo ACLARACIÓN que dice sin ambigüedad que los guisos de proteína como cena NO son `slot_inapropiado` por sí solos; (2) `slot_inapropiado` se restringió a violar la REGLA DURA únicamente, nunca "criterio propio de ligereza"; (3) `nombre_no_corresponde` ganó una nota explícita sobre el caso sutil "mismo tipo de plato, ingrediente NOMBRADO ausente" (ej. moro sin guandules aunque siga siendo moro); (4) `tecnica_impropia` vs `paso_incoherente` se desambiguaron: el primero es UN paso mal aplicado al alimento (incluye alimentos que deben quedar fríos recibiendo calor), el segundo es una CONTRADICCIÓN ENTRE DOS pasos de la misma receta.

### Calibración 2026-08-01

Comando: `python scripts/calibrate_culinary_judge.py` (modelo `deepseek-v4-flash`, sin `--thinking`, `MEALFIT_CULINARY_JUDGE_GUARD=warn` forzado por el script). Corrida final (post-fix de rúbrica, 10 llamadas LLM reales: 5 buenos + 5 mutados):

```
Modelo juez: deepseek-v4-flash  thinking=False  guard=warn  timeout=45s

FP capa1 sobre buenos: 0 (criterio: 0)
FP juez  sobre buenos: 0 de 36 meals = 0.0% (criterio: <5%)
  juez   combo_absurdo            recall 3/4 = 75%
  capa1  estado_imposible         recall 5/5 = 100%
  capa1  ingrediente_huerfano     recall 5/5 = 100%
  juez   nombre_no_corresponde    recall 4/4 = 100%
  capa1  tecnica_impropia         recall 1/1 = 100%
  juez   tecnica_impropia         recall 1/1 = 100%
  capa1  verbo_imposible          recall 5/5 = 100%
  TOTAL capa1  recall 16/16 = 100%
  TOTAL juez   recall 8/9 = 89%

Veredicto: capa1=OK  juez=OK
```

**Veredicto: ambos criterios del spec §6 cumplidos** — capa1 recall 1.00 + 0 FP (contrato F1 re-confirmado en el mismo run que el juez); juez recall 89% (≥0.80) + FP 0% (<5%). **Autoriza la escalada OFF→`warn`** — la decisión de activarla en prod queda para el owner (Task 14, requiere además ≥1 semana de `warn` limpio antes de plantear `block`).

Caveats de la corrida:
- **N pequeño**: el total juez es 9 defectos (4 `combo_absurdo` + 4 `nombre_no_corresponde` + 1 `tecnica_impropia`) sobre 5 planes — cada miss/hit individual mueve el recall agregado ±11 puntos. El único miss de la corrida final (`combo_absurdo`, 3/4) es varianza de muestreo esperada a `temperature=0.1` (no determinista) — una corrida de verificación puntual sobre ese mismo plan lo capturó correctamente en un intento posterior. El margen sobre el floor de 0.80 (89% vs 80%, 1 solo miss de holgura) es estrecho; si se re-calibra en el futuro con más fixtures, vale la pena correr 2-3 veces y promediar antes de decidir la escalada a `block`.
- **Costo real**: 48 llamadas LLM acumuladas en esta sesión de calibración (corrida original + iteración de rúbrica + verificaciones puntuales) costaron **$0.0086** vía `llm_usage_events` (`node='culinary_judge'`, `model='deepseek-v4-flash'`: 164 970 input tokens, 131 328 de ellos cache hit — ~80% del prefix de la rúbrica cacheado por DeepSeek — + 12 605 output tokens). Muy por debajo del estimado "centavos" del brief. Una corrida de 10 llamadas (el tamaño real de este golden set) cuesta una fracción de centavo.
- **`--thinking` no se probó** en esta calibración — la baseline es sin razonamiento extendido (consistente con la decisión general del owner P1-DAYGEN-TIER-MODEL de no usar thinking en nodos de red/apoyo salvo medición explícita).

---

## Enlaces

- Spec de diseño completo: [`docs/superpowers/specs/2026-07-31-culinary-coherence-design.md`](../../docs/superpowers/specs/2026-07-31-culinary-coherence-design.md) (secciones 4-4c documentan Capa 1; 5-6 documentan Capas 2-3).
- Módulo SSOT: [`backend/culinary_coherence.py`](../culinary_coherence.py) — puro, sin env vars/LLM/DB (capa 1). Juez LLM (capa 2) vive en [`backend/graph_orchestrator.py`](../graph_orchestrator.py) (`run_culinary_judge`, `_CULINARY_JUDGE_RUBRIC`).
- Tests: [`test_p1_culinary_contract.py`](../tests/test_p1_culinary_contract.py) (migración + V1/V2/V3 + 3 superficies, catálogo sintético), [`test_p1_culinary_golden.py`](../tests/test_p1_culinary_golden.py) (golden set contra Neon real), [`test_p1_culinary_judge.py`](../tests/test_p1_culinary_judge.py) (juez: schema, knobs, fail-open, integración en review, parser-based — sin llamadas LLM).
- Script de calibración: [`backend/scripts/calibrate_culinary_judge.py`](../scripts/calibrate_culinary_judge.py) — manual, hace llamadas LLM reales, no corre en CI.
- Reports de implementación (Tasks 3-13 de este SDD, decisiones + concerns detallados): `.superpowers/sdd/2026-07-31-culinary-coherence/task-{3,4,5,6,7,8,9,11,12,13}-report.md`.
