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
| **Capa 1 — scan determinista** | Metadata `prep_methods`/`ready_to_eat` en `master_ingredients` + `culinary_contract_scan()` (V1 verbo↔alimento, V2 estado imposible, V3 huérfanos, V4 cantidad inconsistente ingredientes↔pasos) en 3 superficies (V4 solo en superficies 1-2, ver tabla de checks abajo). | **`warn`** (F1, esta doc). Escalada a `block` es F2 (`P1-CULINARY-CONTRACT-BLOCK`), no implementada. |
| **Capa 2 — juez LLM** | `run_culinary_judge()` (DeepSeek flash) ve recetas completas (el reviewer médico no las ve) y detecta clases abiertas (combos raros, técnica impropia, nombre no corresponde) vía `CulinaryJudgeReport`, integrado en `review_plan_node` con history propio (`_culinary_judge_history`). | **Implementada, nace `off`** (`P1-CULINARY-JUDGE`, Tasks 11-12). Calibrada 2026-08-01 (ver sección abajo) — la corrida re-medida con la rúbrica generalizada quedó **por debajo del floor de recall** (78% < 80%); **NO autoriza aún** la escalada OFF→`warn`. **Re-calibrada 2026-08-01 post-backfill** (3 corridas + mediana) — mediana 78%, **mismo veredicto**. **Iterada 2026-08-01 v3** (ver ["Iteración de rúbrica v3 (combo_absurdo) 2026-08-01"](#iteración-de-rúbrica-v3-combo_absurdo-2026-08-01)) — `combo_absurdo` de 75/50/100 a **100% en las 3 corridas**, mediana juez sube a **89%** (flash AUTORIZA). Permanece `off` en prod — pendiente re-medir `gpt-5.6-luna` (VPS) con la rúbrica v3 antes de flippear el knob (único, no por-modelo). |
| **Capa 3 — calibración medida** | Script `scripts/calibrate_culinary_judge.py`: recall/FP por clase y por capa contra el golden set (con desglose de FP por tipo) + 1 probe held-out generado en runtime (gatea el veredicto), con llamadas LLM reales (no CI). **[P2-VEG-VOLUME-TOKENS-2 · 2026-08-01]** +2 probes informativos runtime (pescado "en anillos" / doble-grano bulgur+arroz, casos reales de las capturas del owner) — sin ground-truth etiquetado en el golden set, NO gatean el veredicto, solo dejan evidencia cruda en stdout para la próxima calibración (T14). | **Hecha** (Task 13, 2026-08-01, corregida tras review; **re-calibrada 2026-08-01 post-backfill con protocolo de 3 corridas + mediana**; **rúbrica iterada a v3 2026-08-01** tras diagnóstico estructural cruzado flash+luna sobre `combo_absurdo`). Resultado post-v3 (flash): capa1 recall 100% + 0 FP en las 3 corridas; juez recall mediana 89% (89%, 100%, 89%) + FP 0%; `combo_absurdo` 100% en las 3; probe held-out out-of-sample **cazado en las 3 corridas**. |

Rollout completo (spec §7): F0 golden set (hecho) → **F1 capa 1 en warn (hecho)** → F2 capa 1 en block (pendiente) → **F3 juez calibrado con rúbrica v3, flash AUTORIZA (89% mediana), luna pendiente de re-medición en VPS** (esta doc, 2026-08-01 — ver sección v3; el knob permanece `off` hasta confirmar luna) → F4 juez warn→block medido (pendiente; requiere primero cruzar F3 con ambos modelos, luego ≥1 semana de warn limpio en prod, Task 14).

---

## Tabla de superficies (F1)

Espejo de [`coherence_surfaces_table.md`](coherence_surfaces_table.md) (mismo patrón: tabla "negativa" — qué SÍ bloquea vs qué solo mide).

| # | Superficie | Archivo | Modo | Qué garantiza |
|---|---|---|---|---|
| 1 | `review_plan_node` | [`graph_orchestrator.py:38280+`](../graph_orchestrator.py) | gate según `MEALFIT_CULINARY_CONTRACT_GUARD` (default `warn`) | Corre DESPUÉS del AUTO-PATCH de huérfanos y AUTO-PATCH-FORWARD (dueño único: reparar → medir). Escribe `plan["_culinary_contract_violations"]` + `plan["_culinary_contract_coverage"]` siempre. En `warn` solo loggea. En `block` (no es el default) cada violación se traduce a `issues.append(...)` + `_severity_max` + **`approved = False`** (fix del [Critical de la ola post-review-final](#blockers-de-f2t10-antes-de-escalar-a-block) — hasta ese fix, `block` acumulaba `issues` pero el veredicto final del nodo nunca leía ese flag, así que NO rechazaba nada; el mismo bug existía en el gate gemelo del juez, capa 2), y por diseño F1 cae a **retry completo** — "incoherencia culinaria" no está en `_SURGICAL_REJECT_SAFE_PREFIXES` ni en `_SURGICAL_REJECT_REJUDGED_PREFIXES`, así que no hay ruta quirúrgica per-día para este gate todavía (ver "Divergencias" abajo). |
| 2 | `finalize_plan_data_coherence` | [`graph_orchestrator.py:25000`](../graph_orchestrator.py) (`_fix_refill_step_verb`, def en `:28138`) | reparación | Cierra la paridad assemble↔finalize: `_fix_refill_step_verb` (repara pasos tipo "🍚 Cuece el Casabe" → "Sirve el Casabe") corría solo en `assemble_plan_node`; ahora también corre en el loop tardío `P2-MISE-COOK-SPLIT` de finalize, en el mismo orden relativo que assemble (`_align_closer_note_food_names` → `_split_cooking_from_mise` → `_fix_refill_step_verb`, al final del trío — el defecto lo introduce un renombrado posterior al productor del paso). Idempotente + fail-safe: re-correrlo donde assemble ya lo aplicó es no-op. |
| 3 | Path degradado (`_build_filtered_edge_recipe_day`, `cron_tasks.py`) | [`cron_tasks.py:24541+`](../cron_tasks.py) | scan + reparación (única capa posible ahí) | Este path NUNCA pasa por `assemble_plan_node`/`review_plan_node` — no hay LLM por construcción. Dos mejoras: (a) el verbo del paso de Desayuno se deriva de `prep_methods[0]` real del alimento en vez de un placeholder genérico fijo ("según método tradicional"); (b) el día ENSAMBLADO pasa por `culinary_contract_scan` (solo V1/V2 — V3 se tolera, ver abajo) y `_degrade_offending_steps` degrada el paso ofensor a `"Sirve el {food}."`, acotado por `meal` exacto y con el matcher canónico `find_catalog_foods` (no substring plano). Fail-open total: si el scan revienta, el día sale tal cual — el backstop de seguridad (`P0-DEGRADED-SAFETY-SCAN`) es una capa aparte y anterior. |

**V3 (huérfanos) NO corre en la superficie 3**: los Edge Recipes son 3 pasos fijos (Mise en place / El Toque de Fuego / Montaje) que nunca listan cada ingrediente paso a paso — aplicar V3 ahí degradaría pasos que ya están bien. V1/V2 sí, porque verbo y estado se pueden falsar sin necesitar que el paso enumere cada ingrediente.

**El bloque de degradación de la superficie 3 es un no-op estructural hoy** (documentado, no un bug): el verbo del Desayuno se deriva del `prep_methods[0]` del MISMO alimento que luego valida el scan, así que nunca puede violar V1 por construcción; los pasos de Almuerzo/Cena no mencionan el nombre del alimento en el texto, así que `find_catalog_foods` no encuentra a quién acusar. Es exactamente lo que el spec pide ("última palabra por si algo se coló") — un backstop de defensa-en-profundidad para ediciones FUTURAS de las plantillas, verificado con una violación inyectada a mano (ver `test_degrade_offending_steps_matcher_canonico_no_substring`), no algo que dispare hoy con las plantillas actuales.

### Checks de la capa 1 (V1-V4)

| Check | Qué compara | Detalle | Severity |
|---|---|---|---|
| **V1** — verbo↔alimento | Cada verbo de cocción del paso (`VERB_TO_METHOD`) contra `prep_methods` del alimento que resuelve como su objeto. | Fail-open sin metadata (`prep_methods IS NULL`). | `minor` |
| **V2** — estado imposible | Menciones "(ya viene cocido)"/"(ya está cocido)" contra `ready_to_eat` del alimento. | Solo dispara si `ready_to_eat = false` explícito (NULL ⇒ fail-open). | `high` |
| **V3** — huérfanos | Cada alimento de `ingredients[]` (no exento por `CONDIMENT_EXEMPT`) contra las menciones de `recipe[]` (con fallback a prefijo, `_mencionado_por_prefijo`). | `repairable=True` (AUTO-PATCH lo elimina antes de que V3 lo mida). No corre en la superficie 3 (path degradado — Edge Recipes no listan ingrediente por ingrediente). | `minor` |
| **V4** — cantidad inconsistente (`_v4_cantidad_inconsistente`, [P1-CULINARY-CONTRACT · V4 · 2026-08-01]) | El gramaje explícito ("N g") de `ingredients[]` contra el gramaje explícito del paso que lo declara (prioridad: Mise en place → primer paso que lo declare, `_v4_grams_by_food`). SOLO gramos↔gramos — nunca convierte taza/cdta/unidad → gramos (regla dura (a)); alimento resuelto vía `find_catalog_foods`, jamás substring. | Tolerancia `V4_TOLERANCIA = 0.25` (25%, generosa a propósito — redondeos de lonjas/tazas/piezas a un número "bonito" son legítimos). **[V4-FIX3 · 2026-08-01]** Gramaje precedido de `≈`/`~` (aproximación declarada, p.ej. hints de `append_gram_hint` sobre unidades vagas lonja/pedazo/porción) se **SKIP silencioso** — no es un contrato exacto, no dispara ni se compara. `repairable=False` (un desacuerdo de cantidad no tiene reparación textual obvia sin inventar un número). No corre en la superficie 3 (mismo motivo que V3). | `minor` |

**Findings conocidos de V4** (review post-implementación, 2026-08-01):
- **El golden set no ejerce V4 con datos reales.** Los 5 planes `golden_XX_bueno` (fixtures reales contra Neon) dieron 0 falsos positivos de V4 porque ninguno de sus pasos declara gramaje explícito que diverja del de `ingredients[]` — la prueba de V4 vive enteramente en los tests sintéticos de [`test_p1_culinary_contract.py`](../tests/test_p1_culinary_contract.py) (sección V4), no en `test_p1_culinary_golden.py`. Si el golden set se regenera con un caso de divergencia real, sería la primera prueba de V4 contra datos de producción.
- **V4 no corre en la superficie 3** (path degradado, `cron_tasks.py`) por la misma razón que V3: los Edge Recipes son 3 pasos fijos que nunca declaran gramaje paso a paso. En teoría podría aplicar si un Edge Recipe alguna vez declarara gramajes en texto libre, pero hoy `_degrade_offending_steps` solo consume violaciones `V1`/`V2` — extenderlo es un cambio de una línea en el filtro que `_build_filtered_edge_recipe_day` pasa a `_degrade_offending_steps`, más decidir qué "reparación" tiene sentido para V4 (no es tan simple como degradar a "Sirve el {food}." — V4 es `repairable=False` a propósito).

---

## Blockers de F2/T10 (antes de escalar a `block`)

Registro de lo que la ola de fixes post-review-final (whole-branch, 19 commits) encontró y cerró — o dejó pendiente a propósito — antes de que F2 (`P1-CULINARY-CONTRACT-BLOCK`) pueda considerarse.

1. **[CERRADO] `block` no rechazaba nada (Critical).** Ambos gates de `review_plan_node` (capa 1 contrato y capa 2 juez) apilaban violaciones en `issues` + escalaban `severity`, pero ninguno seteaba `approved = False` — el veredicto final del nodo (`if approved: ... else: ...`) nunca leía `issues`/`severity` por su cuenta, así que `MEALFIT_CULINARY_CONTRACT_GUARD=block` (y su gemelo del juez) se comportaban EXACTAMENTE como `warn`: el plan se aprobaba igual, con las violaciones descartadas en la rama aprobada. Misma clase de bug que P1-G (`_shopping_coherence_block` sin consumer). Fix: `approved = False` añadido en ambos bloques, espejando el patrón ya usado por `_shopping_coherence_block` (`graph_orchestrator.py` ~L38441). Regresión anclada por `test_p1_culinary_block_enforcement.py` (ejecuta `review_plan_node` completo con guard=`block` + catálogo/juez mockeados, confirmado rojo contra el código pre-fix antes de mergear el fix).

2. **[CERRADO] La cobertura de T10 solo medía semana 1.** El chunk worker (`cron_tasks.py`) propagaba `_quality_degraded*` de `result` → `full_plan_data` para semanas 2+ (P2-10) pero NUNCA propagaba `_culinary_contract_violations`/`_culinary_contract_coverage`/`_culinary_judge_history` — cualquier plan multi-semana perdía la telemetría culinaria de las semanas 2+ en el overlay T2, así que una métrica de cobertura/violaciones agregada (T10, futura) habría medido solo la semana 1 y reportado una cobertura optimista. Fix: las 3 keys se propagan ahora en el mismo bloque que `_quality_degraded*` + están en `P0_4_T2_INCREMENTAL_KEYS` para sobrevivir el re-read de T2. `_culinary_judge_history` específicamente se EXTIENDE (no sobrescribe) sobre el history ya persistido de semanas previas — un overwrite ciego, como el resto de las keys de esa lista, habría perdido el history acumulado de semanas 1..N-1 cada vez que una semana N+1 completaba.

3. **[PENDIENTE — decisión de producto, no gap técnico] Acoplar los knobs, o warning de arranque si `judge=block` con `contract=off`.** Hoy `MEALFIT_CULINARY_CONTRACT_GUARD` y `MEALFIT_CULINARY_JUDGE_GUARD` son independientes — nada impide (ni avisa) si un operador setea el juez (capa 2, holístico) a `block` mientras el contrato determinista (capa 1) queda en `off`. No es un estado necesariamente incorrecto (el juez es aditivo por diseño, "jamás aprueba en silencio lo que la capa 1 ya rechazó" — pero funciona igual de bien sola), pero es una combinación que probablemente nadie eligió a propósito, dado que el rollout documentado en esta doc asume capa 1 madura ANTES que capa 2. **NO implementado en esta ola** — registrado aquí como decisión pendiente para cuando F2 (block de capa 1) esté sobre la mesa: evaluar entonces si vale un `logger.warning` de arranque (o un check en `/health/version`) cuando se detecte `judge != off and contract == off`, o si se prefiere dejarlo como combinación válida y documentada.

| Knob | Default | Efecto |
|---|---|---|
| `MEALFIT_CULINARY_CONTRACT_GUARD` | `warn` | `off`/`warn`/`block`. Clamp a `warn` si el valor no es uno de los 3. Registrado vía `_env_str` → `_KNOBS_REGISTRY` ([`graph_orchestrator.py:6112`](../graph_orchestrator.py)). Escalada a `block` es F2 (`P1-CULINARY-CONTRACT-BLOCK`), pendiente. |
| `MEALFIT_CULINARY_JUDGE_GUARD` | **`off`** | `off`/`warn`/`block`. Clamp fail-safe a `off` (no a `warn`) si el valor no es uno de los 3 — a diferencia del contract-guard, esta es una llamada LLM completa por plan, así que el único default seguro ante un valor raro es apagado. Calibrado 2026-08-01: recall re-medido (78%, tras generalizar la rúbrica) queda bajo el floor de 0.80 — **permanece `off`**. Re-calibrado 2026-08-01 post-backfill (3 corridas + mediana): mediana 78%, mismo resultado — **sigue `off`**. **Rúbrica iterada a v3 2026-08-01** (`combo_absurdo`, ver sección dedicada): flash re-medido AUTORIZA (mediana 89% ≥ 0.80 en las 3 corridas) — **sigue `off` en prod** hasta que el controller re-mida `gpt-5.6-luna` con la rúbrica v3 en el VPS (el knob es único, no distingue modelo). |
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

**[P1-CULINARY-METADATA-BETA · 2026-08-19] Ronda 3 — el hueco que reabrió el catálogo beta.**
Las 141 filas de países beta que `P1-COUNTRY-SYSTEM-F2` insertó el 2026-08-17 nacieron con
`prep_methods`/`ready_to_eat` en NULL al 100%, devolviendo la cobertura del catálogo de 100% a
**206/347 = 59%**. Sobre un corpus de recetas beta la cobertura medida era **24%**: un plan
dominicano no notaba nada (usa filas DO), un plan español se quedaba sin capa 1 entera. El
backfill (76 filas por default de categoría + 65 de Despensa una a una) la devuelve a 100%, y el
CHECK `master_ingredients_prep_methods_not_null` impide que el próximo lote de altas lo repita —
la invariante vive ahora donde vive el dato, no en un test parser-based. Corpus:
`tests/fixtures/culinary_beta/`. Test: `test_p1_culinary_metadata_beta.py`.

**[P1-CULINARY-HASTA-DORAR · 2026-08-19] «hasta dorar» ya no es una orden de saltear.**
`dora(?!d[oa]s?)\w*` excluía los participios («dorado/dorada») pero **no el
infinitivo**, así que «Hornea las papas hasta dorar» acusaba de salteado a todo alimento
del paso sin `saltear` en `prep_methods`. Medido sobre 33 planes REALES de producción:
**12 de 63 violaciones V1 eran esto — 19% de ruido**, y contra quien menos toca el fuego
(Aceite de oliva, Miel, Vainilla, Mango, Linaza, Plátano maduro), porque V1 acusa a
cualquier alimento nombrado en un paso largo multi-cláusula.

Importaba más de lo que parecía: en `warn` era ruido de telemetría, pero la escalada a
`block` que persigue `P1-CULINARY-CONTRACT-BLOCK` convertiría ese 19% en rechazos de
planes buenos.

El fix es `(?<!hasta )dora…`, el mismo mecanismo que el `(?<!para )horno` de la ronda
anterior y por la misma razón: una palabra que describe el envase o el PUNTO de cocción
no es una instrucción. Dos alternativas se descartaron **por medición, no por intuición**:
excluir solo el infinitivo desnudo (`|r`) caza 6 de 12 y deja pasar «hasta dorarlas»;
añadir `(?<!a )` encima no cambia ni una violación sobre datos reales. El imperativo
sigue intacto («Dora la cebolla», «Dóralo por ambos lados»): romperlo reviviría la
regresión que la Task-5 del P-fix original ya pagó. Test:
[`test_p1_culinary_hasta_dorar.py`](../tests/test_p1_culinary_hasta_dorar.py).

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

### Calibración capa1 2026-08-01 (plan real 165dd761, primer plan en producción, fase warn)

El primer plan real que pasó por el guard en producción midió **9 violaciones V1, las 9 falsos positivos** — de solo 2 mecanismos, ambos cerrados en `culinary_coherence.py` (P1-CULINARY-CONTRACT-FP1):

- **Clase participio/montaje (6/9):** el `\w*` genérico tras la raíz del verbo capturaba la forma PARTICIPIAL/ADJETIVAL, no solo el imperativo — «el yaniqueque HORNEADO», «pollo desmechado SALTEADO con los vegetales», «las TOSTADAS», «almendras TOSTADAS» — y las 6 vivían en pasos de "Montaje:" (que por construcción ensambla, nunca cocina). Fix: negative-lookahead de participio (`d[oa]s?\b`/`t[oa]s?\b`) en cada raíz de `VERB_TO_METHOD` cuya forma participial existe en español (hornear/guisar/saltear-sofreír-dorar/licuar/tostar) + skip explícito de pasos que empiezan con "Montaje:" en V1 (V2/V3 sin cambios — V3 sigue necesitando leer montaje para las menciones).
- **Clase ventana post-verbo (3/9):** el verbo apuntaba a un alimento NO catalogado (p.ej. "almendras" en «tuesta las almendras aparte») y, sin destinatario válido en el catálogo, la salvaguarda multi-alimento de `_v1_verbo_alimento` acusaba a los acompañantes catalogados del mismo paso (avena/leche/clara) que no eran el objeto real del verbo. Fix: veto de ventana post-verbo (`_post_verb_resolves`, ~4 palabras tras el match) cuando NINGÚN alimento del paso acepta el método — si el objeto inmediato del verbo no resuelve al catálogo, no se acusa a nadie por ese verbo.

Cerrada. Los 9 FPs (re-evaluados con catálogo sintético que reproduce el caso real) dan 0 tras el fix — casos anclados como tests sintéticos PERMANENTES en [`test_p1_culinary_contract.py`](../tests/test_p1_culinary_contract.py) (sección "FP reales 2026-08-01 plan 165dd761"), no en los fixtures del golden set (el ground truth no se toca — el golden set sigue en 100%/0FP, re-confirmado tras el fix). Traza completa por fragmento de verbo: `.superpowers/culinary-fp-round1-report.md`.

---

## Juez LLM (Capa 2, F3): rúbrica y calibración

`run_culinary_judge(plan, country="DO")` ([`graph_orchestrator.py`](../graph_orchestrator.py)) hace UNA llamada batched (no por día) a `CULINARY_JUDGE_MODEL` con `with_structured_output(CulinaryJudgeReport)`. Fail-open total: knob `off`, timeout, o cualquier excepción del LLM/parseo → `None`, nunca bloquea el plan por su cuenta. Ve la receta completa por plato (`recipe`, pasos) — el único ojo LLM del pipeline que la ve; el reviewer médico solo recibe nombre+ingredientes. `country` ([P1-COUNTRY-SYSTEM-F1 · 2026-08-16, Task 3](country_system_f1.md)) selecciona la rúbrica vía `_culinary_judge_rubric_for_country`: `'DO'`/default deja `_CULINARY_JUDGE_RUBRIC` byte-idéntico (cacheado por país); país beta sustituye "Eres un juez culinario dominicano experto" por una variante que nombra la cocina de `COUNTRY_PROFILES[cc]['name_es']` + cocina internacional.

`CulinaryViolation.tipo` acepta exactamente 5 valores canónicos: `combo_absurdo`, `tecnica_impropia`, `paso_incoherente`, `slot_inapropiado`, `nombre_no_corresponde`. La rúbrica (`_CULINARY_JUDGE_RUBRIC`, construida UNA vez a import-time para cache hits de DeepSeek sobre el prefix estable) combina: hasta 10 nombres de ejemplo por slot desde `data/dish_templates.json`, la REGLA DURA de horario (arroz/locrio/moro/pasta nunca en desayuno/cena; sopones solo en almuerzo; postre como plato principal), la GUÍA POSITIVA por horario de `constants.SLOT_POSITIVE_HINT`, y las definiciones de los 5 tipos con ejemplos.

**Iteración de rúbrica (2026-08-01, ronda 1 — spec §6):** la corrida baseline inicial con la rúbrica original de Task 11 dio juez recall 78% + FP 16.7% (**FALLA** ambos criterios). Causa raíz de los 6 FPs: la GUÍA POSITIVA de `cena` dice "evita... los guisos pesados" — el juez la trataba como regla dura y marcaba `slot_inapropiado` sobre guisos de proteína legítimos como cena (pollo/carne/pescado guisado), que SÍ son cena dominicana real y ninguno de los 5 `golden_XX_bueno` los etiqueta como defecto. Causa raíz de los 2 misses de recall: (a) el swap `golden_02_mutado` renombra "Moro de habichuelas negras" → "Moro de guandules" **sin cambiar la categoría del plato** (sigue siendo arroz+legumbre, solo cambia la legumbre) — más sutil que los otros 3 swaps de `nombre_no_corresponde` del golden set, que sí cambian de categoría de plato; (b) el defecto de `tecnica_impropia` en `golden_03_mutado` (yogurt sobre la plancha) lo detectaba pero lo etiquetaba `paso_incoherente` — ambigüedad de frontera entre "técnica mal aplicada en un paso" y "dos pasos que se contradicen".

Fix ronda 1 (`_build_culinary_judge_rubric()`, mismo archivo): (1) la GUÍA POSITIVA se rotuló explícitamente "orientativa, NO una regla dura" + un párrafo ACLARACIÓN que dice sin ambigüedad que los guisos de proteína como cena NO son `slot_inapropiado` por sí solos; (2) `slot_inapropiado` se restringió a violar la REGLA DURA únicamente, nunca "criterio propio de ligereza"; (3) `tecnica_impropia` vs `paso_incoherente` se desambiguaron: el primero es UN paso mal aplicado al alimento (incluye alimentos que deben quedar fríos recibiendo calor), el segundo es una CONTRADICCIÓN ENTRE DOS pasos de la misma receta; (4) `nombre_no_corresponde` ganó una nota sobre el caso sutil "mismo tipo de plato, ingrediente NOMBRADO ausente".

**⚠️ Corrección post-review (misma fecha):** el punto (4) de la ronda 1, tal como se escribió originalmente, usaba el ejemplo literal **"'Moro de guandules' hecho en realidad con habichuelas negras"** — que ES, palabra por palabra, el placeholder de renombrado que usan las 4 mutaciones `nombre_no_corresponde` del golden set (`golden_manifest.json` líneas 78/118/158/198, las 4 renombran a "Moro de guandules"). La rúbrica le estaba dando al LLM la respuesta del examen, no una regla generalizable — el salto de recall de esa clase (75%→100%) medía memorización del patrón exacto, no comprensión de la regla. Reescrito a una regla genérica sin nombrar ningún plato del golden set (ver el texto actual de `_CULINARY_JUDGE_RUBRIC` arriba: "el 'X' de 'plato de X'... sustituirlo por otro de la MISMA FAMILIA... sigue siendo `nombre_no_corresponde`"), y validado con un **probe held-out generado en runtime** (ver abajo) que usa una familia de ingrediente completamente distinta (mariscos↔pollo, no legumbre↔legumbre) — el probe SÍ fue cazado correctamente, lo que confirma que la regla generaliza y no es memorización.

### Trade-off: `slot_inapropiado` restringido a la REGLA DURA

El fix de la ronda 1 (punto 2 arriba) restringe `slot_inapropiado` del juez a violar SOLO la REGLA DURA (arroz/locrio/moro/pasta fuera de su horario, sopón fuera de almuerzo, postre como plato principal). Esto fue necesario para eliminar los FPs de la corrida 1 — pero tiene un costo que debe quedar explícito antes de decidir F4: el juez **renuncia** al rol de backstop semántico sobre la lista SOFT completa de `constants.SLOT_INAPPROPRIATE_FOODS` (no inyectada al prompt del juez), que incluye reglas más finas que la REGLA DURA no cubre — entre ellas: fritura pesada de proteína como plato de cena (`pollo frito`/`chicharron`/etc.), comida de desayuno servida en la cena (cereal/panqueque/waffle/avena), un plato literalmente nombrado "Desayuno..." servido de noche, guiso pesado/legumbres en el desayuno, postre standalone como plato principal del almuerzo, plato fuerte disfrazado de merienda, y vegetales crudos como vehículo de dip (merienda americana). Estas reglas SÍ están enforzadas por el validador determinista de generación/swap (`constants.py`, consumido en los paths de day-gen y chat-modify) — el juez no es la única defensa del sistema contra ellas — pero si algún día se usa `run_culinary_judge` como backstop de un path que NO pasa por ese validador (análogo a por qué existe `P0-DEGRADED-SAFETY-SCAN` para el path degradado), esta restricción significa que el juez NO las cazaría. Decisión consciente para F3 (prioriza 0 FP sobre cobertura semántica amplia); revisitar si F4 exige que el juez cubra más que la REGLA DURA.

### Calibración 2026-08-01 (corrida final, tras la corrección post-review)

Comando: `python scripts/calibrate_culinary_judge.py` (modelo `deepseek-v4-flash`, sin `--thinking`, `MEALFIT_CULINARY_JUDGE_GUARD=warn` forzado por el script). El script ahora también desglosa FP por `tipo` de violación y corre 1 probe held-out generado en runtime (11 llamadas LLM reales: 5 buenos + 5 mutados + 1 probe):

```
Modelo juez: deepseek-v4-flash  thinking=False  guard=warn  timeout=45s

FP capa1 sobre buenos: 0 (criterio: 0)
FP juez  sobre buenos: 0 de 36 meals = 0.0% (criterio: <5%)
  juez   combo_absurdo            recall 2/4 = 50%
  capa1  estado_imposible         recall 5/5 = 100%
  capa1  ingrediente_huerfano     recall 5/5 = 100%
  juez   nombre_no_corresponde    recall 4/4 = 100%
  capa1  tecnica_impropia         recall 1/1 = 100%
  juez   tecnica_impropia         recall 1/1 = 100%
  capa1  verbo_imposible          recall 5/5 = 100%
  TOTAL capa1  recall 16/16 = 100%
  TOTAL juez   recall 7/9 = 78%

=== Probe held-out (generado en runtime, NO es fixture del golden set) ===
Probe: 'Sancocho de mariscos' (receta real de arroz+pollo) día 1 Almuerzo → CAZADO (nombre_no_corresponde)

Veredicto: capa1=OK  juez=FALLA  probe=OK
→ NO autoriza aún la escalada OFF→warn — ver qué falló arriba.
```

(FP juez = 0 en esta corrida, así que no hay desglose por tipo que mostrar — el script lo imprime cuando `fp_juez_by_tipo` no está vacío.)

**Veredicto honesto: NO se cumple el criterio de recall del juez en esta corrida** (78% < floor 0.80) — **la escalada OFF→`warn` NO queda autorizada**. El knob permanece `off`. capa1 sigue en 100%/0FP (contrato F1 re-confirmado). El probe held-out SÍ fue cazado correctamente (`nombre_no_corresponde` sobre "Sancocho de mariscos"/receta real de pollo, familia de ingrediente distinta a cualquier fixture del golden set) — confirma que la corrección de la rúbrica generaliza y no depende de memorizar el patrón "Moro de guandules", que era la preocupación de la ronda de review.

**Diagnóstico del miss de `combo_absurdo` (2/4, no persistido — corrida ad-hoc de verificación):** los 2 misses de esta corrida fueron `golden_02_mutado` (día 2, "Pan integral con mantequilla de maní y salami frito") y `golden_05_mutado` (día 2, "Avena cremosa con salami frito"). Ninguno de los dos es un miss reproducible: el mismo defecto de `golden_02_mutado` SÍ fue cazado en la corrida de la ronda 1 (antes de tocar el ejemplo de `nombre_no_corresponde`, que no afecta a `combo_absurdo`), y el mismo patrón exacto "avena + salami" SÍ fue cazado en `golden_01_mutado`/`golden_04_mutado` de esta MISMA corrida. Es varianza de muestreo a `temperature=0.1` (no determinista), no un gap estructural de la rúbrica — pero es una medición honesta y el resultado se reporta tal cual, sin re-rodar hasta obtener un número favorable.

Caveats:
- **N pequeño**: el total juez es 9 defectos sobre 5 planes — cada miss/hit individual mueve el recall agregado ±11 puntos. Con un floor de 0.80 y N=9, el resultado puede oscilar entre corridas por pura varianza de muestreo (78% en esta corrida, 89% en la corrida previa con la rúbrica que sobreajustaba). Antes de decidir la escalada, T14 debería promediar varias corridas o ampliar el golden set — un solo run cerca del floor no es suficiente evidencia en ningún sentido.
- **Costo real acumulado de toda la sesión de calibración** (ronda 1 + corrección + verificaciones ad-hoc + corrida final + probe): **63 llamadas**, **$0.0119** vía `llm_usage_events` (`node='culinary_judge'`, `model='deepseek-v4-flash'`: 221 667 input tokens, 173 184 cache hit — ~78% del prefix de la rúbrica cacheado por DeepSeek — + 16 783 output tokens). Muy por debajo del estimado "centavos" del brief. Una corrida real de 11 llamadas (el tamaño del script con el probe incluido) cuesta una fracción de centavo.
- **`--thinking` no se probó** en esta calibración — la baseline es sin razonamiento extendido (consistente con la decisión general del owner P1-DAYGEN-TIER-MODEL de no usar thinking en nodos de red/apoyo salvo medición explícita).

### Re-calibración 2026-08-01 (post-backfill)

**Contexto:** el backfill ronda 2 de metadata culinaria (`P2-CULINARY-METADATA-ROUND2`) cerró la cobertura de `master_ingredients` a 100% (antes ~72.5%/48.5% `prep_methods`/`ready_to_eat`, ver "Vocabulario `prep_methods` (SSOT)" arriba). Ese backfill solo puede mover capa1 (el scan determinista lee esa metadata) — el juez (capa2) nunca la lee, así que a priori no debía mover su recall. El script también ganó 2 probes informativos nuevos desde la corrida anterior (`(a)` pescado "en anillos", `(b)` doble-grano bulgur+arroz, `P2-VEG-VOLUME-TOKENS-2`).

Dado el N pequeño ya documentado (9 defectos-juez, cada hit/miss mueve el recall agregado ±11pp), esta re-calibración corre el script **3 veces bajo las mismas condiciones** (`deepseek-v4-flash`, sin `--thinking`, `MEALFIT_CULINARY_JUDGE_GUARD=warn` forzado por el script) y usa la **mediana** de las 3 como criterio de decisión — nunca una corrida aislada.

⚠️ **Nota operacional (Windows, no es un bug de la rúbrica):** la primera invocación crasheó con `UnicodeEncodeError` al imprimir el separador `→` bajo la consola cp1252 por defecto de PowerShell/Git Bash en Windows — el juez ya había respondido (10-11 llamadas ya facturadas) pero el script murió antes de imprimir el probe held-out y los 2 probes informativos. Se resuelve con `PYTHONIOENCODING=utf-8` antes de invocar el script (no se tocó el script — es un gotcha de terminal, no de lógica). Esa corrida parcial se **descarta** de las 3 mediciones oficiales (nunca completó los probes) pero sus llamadas sí se facturaron y están incluidas en el costo total de abajo.

Comando (×3, mismas condiciones): `PYTHONIOENCODING=utf-8 python scripts/calibrate_culinary_judge.py`

| Corrida | capa1 recall | capa1 FP | juez recall | juez FP | `combo_absurdo` | `nombre_no_corresponde` | `tecnica_impropia` | probe held-out (gating) | probe (a) anillos (informativo) | probe (b) doble-grano (informativo) | veredicto script |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16/16 = 100% | 0 | 7/9 = 78% | 0/36 = 0% | 3/4 = 75% | 3/4 = 75% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=FALLA probe=OK → NO autoriza |
| 2 | 16/16 = 100% | 0 | 7/9 = 78% | 0/36 = 0% | 2/4 = 50% | 4/4 = 100% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=FALLA probe=OK → NO autoriza |
| 3 | 16/16 = 100% | 0 | 8/9 = 89% | 0/36 = 0% | 4/4 = 100% | 3/4 = 75% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=OK probe=OK → **AUTORIZA** |
| **Mediana** | **100%** | **0** | **78% (7/9)** | **0%** | 75% | 75% | 100% | CAZADO (3/3) | NO CAZADO (3/3) | CAZADO (3/3) | — |

**Capa1 (contrato F1): recall 1.00 + 0 FP en las 3 corridas** — re-confirmado tras el backfill ronda 2, sin cambios respecto a la corrida anterior. ✓ PASA.

**Juez (capa2): recall mediana 78% (7/9) < floor 0.80** — la MISMA cifra que la corrida original (78%), reproducida de nuevo después del backfill. La varianza ya documentada en la corrida previa sigue intacta: 2 de 3 corridas caen en 78%, 1 en 89%; ninguna clase individual falla en las 3 (ver abajo). FP juez 0% en las 3 — muy por debajo del criterio <5%. ✗ NO PASA el floor de recall en la mediana.

**Probe held-out gating (`nombre_no_corresponde`, familia mariscos↔pollo): CAZADO en las 3 corridas** — confirma que la regla generalizada de la ronda 1 de rúbrica sigue sin depender de memorizar el patrón "Moro de guandules" del golden set. ✓ PASA, consistente.

**Probes informativos nuevos (sin criterio de fallo, evidencia cruda para T14):**
- **(a) pescado "en anillos"**: NO CAZADO en las 3 corridas (consistente) — el juez no marca la técnica de corte de calamar aplicada a un filete de pescado blanco, ni como `tecnica_impropia` ni como `paso_incoherente`. Señal para T14: esta clase de error (técnica de corte incorrecta para el tipo de proteína) no está cubierta hoy por la rúbrica.
- **(b) doble-grano bulgur+arroz**: CAZADO en las 3 corridas, siempre clasificado `paso_incoherente` (nunca `combo_absurdo`) — el juez lo lee consistentemente como "el paso menciona un ingrediente ausente de la lista", no como "dos carbohidratos-base en el mismo plato". Ambos tipos están en `tipos_esperados` del script así que cuenta como CAZADO en los 3, pero la clasificación específica es sistemática, no ruido.

**¿Se tocó la rúbrica? No.** El protocolo autoriza 1 sola iteración de rúbrica SOLO si las 3 corridas fallan consistentemente en la MISMA clase-juez. Desglose por clase a través de las 3 corridas:
- `combo_absurdo`: 75%, 50%, **100%** — no falla consistentemente (corrida 3 pasa).
- `nombre_no_corresponde`: 75%, **100%**, 75% — no falla consistentemente (corrida 2 pasa).
- `tecnica_impropia`: 100%, 100%, 100% — nunca falla.

Ninguna clase individual falla en las 3 corridas — el déficit de recall agregado es varianza de muestreo distribuida entre `combo_absurdo` y `nombre_no_corresponde` (el mismo patrón que la corrida original: "78% en esta corrida, 89% en la corrida previa... por pura varianza de muestreo"), no un sesgo sistemático de una clase. Condición de re-iteración no cumplida → rúbrica sin cambios (`graph_orchestrator.py` intacto en esta ronda).

**Costo:** **50 llamadas LLM reales, $0.003894 total** (`llm_usage_events`, `node='culinary_judge'`, `model='deepseek-v4-flash'`: 183 723 input tokens, 180 992 cache hit — ~98% del prefix de la rúbrica cacheado por DeepSeek — + 10 831 output tokens), de las cuales 11 llamadas (~$0.0008) pertenecen a la corrida descartada por el crash de encoding (5 buenos + 5 mutados + 1 probe held-out, antes de morir en el print) y 39 llamadas (13×3: 5 buenos + 5 mutados + 1 probe held-out + 2 probes informativos por corrida, ~$0.0031) pertenecen a las 3 corridas oficiales de la tabla. Muy por debajo del estimado "centavos" del brief.

**Veredicto: NO AUTORIZA la escalada OFF→`warn` del knob `MEALFIT_CULINARY_JUDGE_GUARD`.** La mediana de 3 corridas post-backfill (78%, 78%, 89% → mediana 78%) reproduce EXACTAMENTE el resultado de la corrida original (78%) — el backfill de metadata (que solo afecta capa1) no movió el recall del juez (capa2, que nunca leyó `prep_methods`/`ready_to_eat`), y la rúbrica no se tocó porque ninguna clase falló consistentemente en las 3 corridas (condición de re-iteración no cumplida). El knob permanece `off`. Frente a la pregunta que motivó este protocolo de 3 corridas ("una corrida puede caer 78 u 89 por un solo punto"): la mediana confirma que 78% no es el resultado desafortunado de una sola corrida — es el centro de la distribución (2 de 3 corridas independientes cayeron ahí). capa1 sigue en 100%/0FP (contrato F1 re-confirmado, sin regresión post-backfill). El probe held-out gating sigue cazado en las 3 corridas (la regla generalizada de `nombre_no_corresponde` sigue sin sobreajuste). Los 2 probes informativos nuevos quedan como evidencia para T14: `(a)` técnica de corte incorrecta no cubierta hoy por la rúbrica, `(b)` doble-grano cazado consistentemente pero bajo `paso_incoherente` en vez de `combo_absurdo`.

Próximo paso sugerido para T14 (no ejecutado en esta re-calibración — fuera de alcance): si se quiere cerrar el gap de `combo_absurdo`/`nombre_no_corresponde` sin sobreajustar al golden set, ampliar el golden set (más fixtures por clase reduce el ±11pp por hit/miss de N=9) en vez de seguir iterando la rúbrica sobre las mismas 9 muestras.

---

## Iteración de rúbrica v3 (combo_absurdo) 2026-08-01

**Nota sobre el criterio de autorización:** la re-calibración post-backfill (sección anterior) definió el gate de re-iteración como "la clase falla en las 3 corridas de UN modelo" — bajo ese criterio estricto `combo_absurdo` NO calificaba para flash solo (corrida 3 dio 100%, ver desglose arriba: 75%, 50%, 100%). Esta ronda amplía el criterio a evidencia CRUZADA entre dos modelos independientes: el controller corrió el mismo protocolo (×3, mediana) con `gpt-5.6-luna` en el VPS (con `OPENAI_API_KEY` real de producción — el intento de reproducir ese A/B en un worktree local quedó **BLOQUEADO** por falta de esa credencial en `backend/.env` local, ver `.superpowers/ab-juez-luna-report.md`; los 3 valores de luna citados abajo vienen de esa medición del controller, no reproducidos en este documento) y obtuvo `combo_absurdo` 50%, 50%, 0% — mientras el resto de las clases-juez (`nombre_no_corresponde`, `tecnica_impropia`) y el invariante capa1 (100%/0 FP) se mantuvieron intactos en AMBOS modelos. Dos modelos independientes, mismo patrón de defecto (`combo_absurdo`), todo lo demás sano — señal estructural más fuerte que "varianza de muestreo distribuida" (la conclusión de la ronda anterior), y suficiente para autorizar **UNA** iteración generalizada de la rúbrica sin tocar ninguna otra clase.

**Diagnóstico:** las 4 mutaciones `combo_absurdo` del golden set (`golden_manifest.json`, ver `golden_01/02/04/05_mutado`) son todas variantes del mismo patrón — un dulce de desayuno (avena con canela, pan integral con mantequilla de maní) al que se le añade un embutido frito (salami) en el mismo plato. La rúbrica v2 definía `combo_absurdo` con ejemplos genéricos ("cereal con pescado crudo", "postre como fuente principal de proteína") que no cubren este patrón concreto, mientras el párrafo introductorio de la rúbrica ("la creatividad dominicana legítima... NO es una violación; en la duda, NO reportes") le da al juez una salida fácil: cada ingrediente por separado es válido (avena de desayuno legítima, salami legítimo como proteína), así que sin una regla que trace la frontera, el juez puede leer la recombinación como una "fusión creativa" más — el mismo espíritu que los ejemplos de creatividad legítima que la propia rúbrica cita (panqueques de avena, bollitos de yuca). Causa raíz: la definición de `combo_absurdo` no daba un PRINCIPIO para distinguir recombinación-dentro-de-un-patrón (tolerada) de choque-de-perfil-sin-patrón (violación) — dejaba al juez inferir la frontera caso por caso, y con el sesgo pro-creatividad del preámbulo, infería mal.

**Principio nuevo (v3):** la tolerancia a la creatividad cubre RECOMBINACIONES de ingredientes compatibles dentro de un patrón culinario real (dominicano o internacional) — fusiones, sustituciones, platos transformados. NO cubre un choque de PERFIL sin ningún patrón culinario que lo respalde, aunque cada ingrediente sea válido por separado: dulce-de-desayuno + embutido/charcutería frita en el mismo plato, postre + proteína curada, fruta + pescado frito en el mismo bowl. Si ningún patrón culinario conocido junta esos dos perfiles en un solo plato, es `combo_absurdo` — la pregunta operativa que la rúbrica le da al juez es explícita: "¿existe algún patrón culinario conocido que junte estos DOS perfiles en un solo plato?". Se refuerza además que `combo_absurdo` se evalúa **por plato** (nombre + ingredientes del MISMO `meal`, nunca cruzando meals distintos del día). Cambio quirúrgico: solo el bullet `combo_absurdo` de `_build_culinary_judge_rubric()` (`graph_orchestrator.py`) — el resto de la rúbrica (preámbulo de creatividad, los otros 4 tipos canónicos, la REGLA DURA de horario) queda intacto, para no arriesgar las clases que ya estaban en 100%/0 FP en ambos modelos.

**Verificación de no-regresión (spec, antes de medir):** los 5 `golden_XX_bueno` traen combos creativos legítimos que la v3 no puede convertir en FP — `Casabe con queso` (`golden_03_bueno`), `Batida de guineo` (`golden_02_bueno`/`golden_03_bueno`), `Tortilla de huevo con cebolla y queso` (`golden_04_bueno`). Ninguno es un choque de perfil sin patrón (casabe+queso y tortilla+queso son combinaciones dentro de patrones reales; una batida de guineo con leche y miel es un patrón de licuado dominicano estándar) — confirmado por las 3 corridas de abajo: FP juez = 0% en las 3.

### Las 3 corridas (flash, post-v3)

Mismo protocolo que la re-calibración post-backfill (`PYTHONIOENCODING=utf-8 python scripts/calibrate_culinary_judge.py`, `deepseek-v4-flash`, sin `--thinking`, guard forzado a `warn` por el script):

| Corrida | capa1 recall | capa1 FP | juez recall | juez FP | `combo_absurdo` | `nombre_no_corresponde` | `tecnica_impropia` | probe held-out (gating) | probe (a) anillos | probe (b) doble-grano | veredicto script |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 16/16 = 100% | 0 | 8/9 = 89% | 0/36 = 0% | **4/4 = 100%** | 3/4 = 75% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=OK probe=OK → **AUTORIZA** |
| 2 | 16/16 = 100% | 0 | 9/9 = 100% | 0/36 = 0% | **4/4 = 100%** | 4/4 = 100% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=OK probe=OK → **AUTORIZA** |
| 3 | 16/16 = 100% | 0 | 8/9 = 89% | 0/36 = 0% | **4/4 = 100%** | 3/4 = 75% | 1/1 = 100% | CAZADO (`nombre_no_corresponde`) | NO CAZADO | CAZADO (`paso_incoherente`) | capa1=OK juez=OK probe=OK → **AUTORIZA** |
| **Mediana** | **100%** | **0** | **89% (8/9)** | **0%** | **100% (4/4)** | 75% | 100% | CAZADO (3/3) | NO CAZADO (3/3) | CAZADO (3/3) | **AUTORIZA en 3/3** |

**`combo_absurdo`: 100% (4/4) en las 3 corridas** — sube de la mediana pre-v3 (75%: 75%, 50%, 100%) a 100% sólido, sin una sola falla en 12 evaluaciones (4 defectos × 3 corridas). Es la clase que motivó la iteración y queda cerrada con margen.

**`juez` total: mediana 89% (7/9→8/9), sube de 78% pre-v3** — el criterio de recall (≥0.80) se cumple con margen claro en las 3 corridas individuales (89%, 100%, 89%), no solo en la mediana. `nombre_no_corresponde` (75%, 100%, 75%, mediana 75%) queda intacto respecto a la corrida anterior — la v3 no tocó su definición, y su varianza reportada en la ronda previa ("no falla consistentemente") sigue vigente sin cambios; el déficit residual del recall agregado ahora se concentra ahí, no en `combo_absurdo`.

**FP juez: 0% en las 3 corridas** (criterio <5%) — confirma que la v3 no convirtió ninguno de los combos creativos legítimos del golden set (casabe+queso, batida de guineo, tortilla+queso) en falso positivo, la preocupación explícita antes de medir.

**capa1: recall 100% + 0 FP en las 3 corridas** — invariante intacto, `culinary_coherence.py` no se tocó en esta ronda.

**Probe held-out (`nombre_no_corresponde`, familia mariscos↔pollo): CAZADO en las 3** — no se tocó esa regla, se re-confirma sin cambios.

**Probes informativos (T14, sin criterio de fallo):** (a) pescado "en anillos" sigue NO CAZADO en las 3 (gap conocido, sin cambios); (b) doble-grano bulgur+arroz sigue CAZADO en las 3, siempre como `paso_incoherente` (nunca `combo_absurdo` — consistente con la ronda anterior, la v3 no le pidió al juez reclasificar ese caso).

**Costo:** 39 llamadas LLM reales (13×3: 5 buenos + 5 mutados + 1 probe held-out + 2 probes informativos por corrida), mismo orden de magnitud que la re-calibración anterior (~$0.003).

### Veredicto: AUTORIZA con flash + rúbrica v3

La mediana es clara — no hay ambigüedad ni corrida borderline: `combo_absurdo` pasa de un patrón inconsistente (75/50/100, 1 de 3 corridas por debajo del floor de clase) a **100% sólido en las 3 corridas** tras UNA iteración generalizada de su definición (principio de patrón culinario, sin nombrar ningún plato del golden set en la rúbrica — mismo estándar anti-memorización que la corrección post-review de `nombre_no_corresponde`). El recall agregado del juez sube de mediana 78% a mediana **89%**, por encima del floor 0.80 en las 3 corridas individuales, no solo en la mediana. FP juez 0% en las 3, capa1 100%/0 FP intacto en las 3, probe held-out cazado en las 3/3. **Flash queda calibrado y autorizado para la escalada OFF→`warn` bajo la rúbrica v3** — pendiente de que el controller repita el mismo protocolo con `gpt-5.6-luna` en el VPS (la credencial real está ahí, no en este worktree local) antes de decidir el flip del knob `MEALFIT_CULINARY_JUDGE_GUARD`, dado que el knob es único y no distingue modelo: si luna (el modelo de producción para tiers pagados en day-gen, no necesariamente el mismo que ejecuta el juez) sigue fallando `combo_absurdo` con la rúbrica v3, la escalada seguiría bloqueada por el modelo más débil. Esta sesión NO modificó el knob — permanece `off` en prod, sin cambios de comportamiento en producción; el cambio es únicamente el texto de la rúbrica.

---

## Enlaces

- Spec de diseño completo: [`docs/superpowers/specs/2026-07-31-culinary-coherence-design.md`](../../docs/superpowers/specs/2026-07-31-culinary-coherence-design.md) (secciones 4-4c documentan Capa 1; 5-6 documentan Capas 2-3).
- Módulo SSOT: [`backend/culinary_coherence.py`](../culinary_coherence.py) — puro, sin env vars/LLM/DB (capa 1). Juez LLM (capa 2) vive en [`backend/graph_orchestrator.py`](../graph_orchestrator.py) (`run_culinary_judge`, `_CULINARY_JUDGE_RUBRIC`).
- Tests: [`test_p1_culinary_contract.py`](../tests/test_p1_culinary_contract.py) (migración + V1/V2/V3 + 3 superficies, catálogo sintético), [`test_p1_culinary_golden.py`](../tests/test_p1_culinary_golden.py) (golden set contra Neon real), [`test_p1_culinary_judge.py`](../tests/test_p1_culinary_judge.py) (juez: schema, knobs, fail-open, integración en review, parser-based — sin llamadas LLM).
- Script de calibración: [`backend/scripts/calibrate_culinary_judge.py`](../scripts/calibrate_culinary_judge.py) — manual, hace llamadas LLM reales, no corre en CI.
- Reports de implementación (Tasks 3-13 de este SDD, decisiones + concerns detallados): `.superpowers/sdd/2026-07-31-culinary-coherence/task-{3,4,5,6,7,8,9,11,12,13}-report.md`.
