# Decisiones de enforcement clínico (gap-audit 2026-06-15)

Dos gaps del audit (G9, G18) NO son deuda técnica sino **decisiones de producto/clínica**: cerrarlas
"implementando" un techo duro o un gate de calidad introduce riesgo (recortar proteína/calorías, perder el
plan real) que requiere validación humana + benchmark. Se documentan aquí (análogo al patrón "Decisiones de
producto" de CLAUDE.md) y se anclan con [`test_p2_clinical_enforcement_decisions.py`](../tests/test_p2_clinical_enforcement_decisions.py)
para que un futuro refactor no asuma un comportamiento distinto sin reabrir la decisión. La opción de
enforcement duro queda como **follow-up explícito**, no como gap silencioso.

---

## G9 — HTA (sodio) y dislipidemia (grasa saturada): enforcement por SUSTITUCIÓN + advisory, NO techo duro

**Decisión:** HTA y dislipidemia se enforzan mediante **sustitución determinista de ingredientes nombrados**
(embutidos/cubitos → fresco para sodio; mantequilla/lácteos enteros/tocino → magro para grasa saturada),
vía `_apply_condition_substitutions` (graph_orchestrator) + `CONDITION_RULES` (condition_rules.py,
`_HTA_SODIUM_SUBS` / `_DYSLIPIDEMIA_SATFAT_SUBS`, P4-DYSLIPIDEMIA-ENFORCED). Adicionalmente, el panel de
micros (micronutrients.py) reporta sodio/satfat vs el techo como **advisory**, y desde gap-audit G5 el techo
es **coverage-aware** (`estimado_alto` cuando hay dato NULL relevante → ya no enmascara violaciones).

**Lo que NO se hace (deliberado):** NO hay un trim cuantitativo que recorte porciones hasta forzar el plan
bajo el techo de sodio/satfat (a diferencia del **cap renal**, que sí trima — ver abajo).

**Por qué:**
- El cap renal de proteína es el ÚNICO trim duro porque la ERC es el riesgo iatrogénico más alto y el
  exceso de proteína es directamente dañino. Para HTA/dislipidemia, el riesgo es crónico-acumulativo, no
  agudo: la sustitución de los ofensores nombrados + el advisory capturan el grueso sin el riesgo de un trim
  ciego.
- Un trim cuantitativo de sodio/satfat recortaría porciones de fuentes que también aportan proteína/calorías
  (queso, carnes) → degradaría la precisión de macros (el sumidero que G4 acaba de cerrar) y podría producir
  porciones no-cocinables. Hacerlo bien exige re-balancear macros tras el trim + validación con benchmark
  (diferido) + idealmente revisión de nutricionista.
- La sal "al gusto" no se modela (no es un ingrediente con cantidad) — un techo duro de sodio daría una falsa
  sensación de precisión sobre un dato que el plan no controla.

**Follow-up (si se decide enforcement duro):** prototipar un `SatFatCeilingConstraint` / `SodiumCeilingConstraint`
en el `ClinicalConstraintEngine` (análogo a `RenalProteinCapConstraint`) que trime + re-balancee + re-cuantice,
validado con `benchmark_macro_compliance.py` sobre perfiles HTA/dislipidemia. Cruza con G11 (motor declarativo).

---

## G18 — Planes que fallan review médico NO-crítico: se ENTREGAN con banner ámbar, NO se reemplazan por fallback

**Decisión:** cuando `review_plan_node` rechaza con severidad NO-crítica (minor, o high-regenerable agotado
por max_attempts/budget), `should_retry` entrega el plan LLM real marcado con `_review_failed_but_delivered` +
`_quality_degraded` (banner ámbar user-facing) + `_emit_plan_quality_degraded_alert` (SRE). Los rechazos
**CRÍTICOS** (alérgeno/schema/renal) SÍ caen al fallback matemático seguro (`needs_critical_fallback` →
`_get_extreme_fallback_plan` con tokens restringidos allergen-aware).

**Lo que NO se hace (deliberado):** NO hay un umbral de calidad bajo el cual un rechazo no-crítico se reemplace
por el fallback matemático genérico.

**Por qué:**
- El fallback matemático es genérico (menú de contingencia sin personalización/variedad/recetas criollas). Para
  un fallo de CALIDAD no-crítico (variedad, fidelidad de skeleton, coherencia), el plan LLM real —aún
  imperfecto— es más útil y personalizado que el fallback. Forzar fallback en cada miss menor degradaría la UX
  para problemas que no son de seguridad.
- La honestidad ya está cubierta: el banner ámbar (`_review_failed_but_delivered`) + el **band-score gate**
  (gap-audit G6, ahora ON) marcan el plan como degradado cuando la precisión medida es baja → el usuario sabe.
- El daño AGUDO ya está cubierto: alérgeno/schema/renal son críticos → fallback seguro. Solo lo no-agudo se
  entrega con disclaimer.

**Follow-up (si se decide un umbral de fallback):** definir, sobre la distribución observable de
`clinical_band_score` (ya persistida por G6) + `holistic_score`, un umbral bajo el cual las ramas
budget/max_attempts prefieran el fallback determinista (que sí pasa FS1-FS9) en vez del plan LLM degradado.
Requiere datos de la distribución en prod (no inventar el umbral).

---

## G11 — Clasificación de las 10 condiciones: engine-enforced vs advisory (motor declarativo a medias, por diseño)

**Decisión:** cada `ConditionRule` (`condition_rules.CONDITION_RULES`) se clasifica por su mecanismo de
enforcement, y el `ClinicalConstraintEngine._REGISTRY` cubre exactamente las engine-enforced. Anclado por
[`test_p2_constraint_engine_parity.py`](../tests/test_p2_constraint_engine_parity.py) (una regla NUEVA sin
clasificar FALLA el test → fuerza una decisión consciente, no queda como advisory silenciosa).

| Condición(es) | Mecanismo | Constraint del engine |
|---|---|---|
| `renal` | Cap de proteína 0.8 g/kg (trim per-comida + red de salida) | `RenalProteinCapConstraint` (id=`renal`) |
| `dm2`, `hta`, `dyslipidemia` | **Sustitución** de tokens ofensores (azúcar→stevia; embutidos→fresco; satfat→magro) | `SubstitutionEngineConstraint` (id=`substitutions`, un solo pase sobre todas) |
| `anemia`, `pregnancy`, `hypothyroid`, `gout`, `nafld`, `pcos` | **Advisory**: prompt_block + panel de micros + (anemia) piso de hierro RDA + (pregnancy) gate fail-hard anti-déficit. NO reescriben porciones. | — (ninguno, por decisión) |

**Lo que NO se hace (deliberado):** los objetivos CUANTITATIVOS de las condiciones advisory NO se modelan
como constraints duros que reescriban el plan. En concreto: **DM2 fibra** (≥14 g/1000 kcal, ADA), **DASH
potasio/magnesio** (HTA) y **techo de satfat** (dislipidemia, ver G9) son advisory de panel + prompt, NO
fuerzan retry ni añaden/recortan ingredientes.

**Por qué:** un `FiberFloorConstraint` que AÑADA leguminosa/avena cuando la fibra cae bajo el piso (análogo
al protein-closer) reescribe el plan → puede chocar con el solver de macros (sube carbos), la variedad y otras
condiciones comórbidas; hacerlo bien exige re-balanceo + validación con benchmark (diferido) + idealmente
revisión de nutricionista. Hasta entonces, el patrón sub-based + advisory es el enforcement honesto y de
menor riesgo. La cautela es la misma del audit clínico P1 (no enforcement clínico cuantitativo sin revisión
humana).

**Follow-up (siguiente nivel del "motor que reescribe"):** migrar DM2-fibra a un `FiberFloorConstraint` en
`_REGISTRY` (el caso límite más claro de objetivo numérico ADA), validado con benchmark sobre perfiles DM2.
Cruza con G9 (`SatFatCeilingConstraint`). El test de paridad ya está listo para exigir su cobertura cuando se
reclasifique de advisory a hard.
