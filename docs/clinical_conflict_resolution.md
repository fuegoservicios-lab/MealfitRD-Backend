# Resolución de conflictos inter-motor (CONDICIONES ↔ FÁRMACOS)

[P2-CLINICAL-CONFLICT-DOC · 2026-06-19] (audit fresco 2026-06-19b, cluster S1)

El generador de planes corre dos motores deterministas en paralelo:

- **CONDICIONES** — `condition_rules.py` (`CONDITION_RULES`) + el panel de micros (`micronutrients.py`) + el `ClinicalConstraintEngine` (`clinical_constraints.py`). Modela ERC, DM2, HTA, dislipidemia, embarazo, etc.
- **FÁRMACOS** — `medication_rules.py` (`MEDICATION_RULES`) + Guard 8d (`graph_orchestrator.py::_apply_deterministic_clinical_layer`). Modela warfarina, IECA/ARA-II, ahorradores-K, metformina, levotiroxina, etc.

Históricamente se componían de forma **aditiva ciega**: los `prompt_block` se concatenaban y el panel de micros no recibía `medications` → una CONDICIÓN podía elevar un micronutriente en la dirección que un FÁRMACO co-presente contraindica. Este doc es el **mapa SSOT** de dónde se reconcilia cada conflicto conocido (la "capa de árbitro" — no es un módulo único sino un conjunto de defensas coordinadas, documentadas aquí para que un refactor no reabra la clase).

> **Diseño (decisión, no deuda):** NO existe un framework genérico de árbitro de N×M reglas. Los conflictos clínicamente accionables son pocos y de baja cardinalidad; cada uno se resuelve en su surface natural (el target del panel, la directiva del prompt, el monitor consumido). Construir un motor genérico violaría la convención "no diseñar para requisitos hipotéticos". Cuando aparezca un conflicto nuevo, se añade una fila a esta tabla + su defensa puntual.

## Tabla de conflictos y su resolución

| # | Conflicto | Dirección segura | Resolución DETERMINISTA (target/panel) | Resolución PROMPT (al generador) | Marker(s) |
|---|---|---|---|---|---|
| C1 | HTA (DASH) sube potasio ↔ ahorrador-K / IECA-ARA-II lo elevan en sangre | Moderar potasio (hiperkalemia → arritmia) | El panel NO eleva el piso DASH-K (4700) ni emite la nota "come más guineo" cuando hay fármaco-K: `build_micronutrient_report(k_elevating_med=…)` gateado por `detect_potassium_elevating_med` | `prompt_block` del ahorrador-K lleva "PRECEDENCIA sobre … AUMENTAR el potasio … incluida DASH" | `P1-POTASSIUM-PANEL-MED-AWARE`, `P1-POTASSIUM-SPARING-DIURETIC` |
| C2 | HTA (DASH) sube potasio ↔ ERC lo restringe | Moderar potasio | El panel NO eleva el piso DASH-K si `_has_renal` (`micronutrients.py`); el cap renal KDIGO enforced | Árbitro `hta+renal` en `build_condition_prompt` (modera potasio/leguminosas, conserva sodio bajo) | `P2-RENAL-HTA-POTASSIUM-GUARD`, `P2-RENAL-HTA-POTASSIUM-PROMPT` |
| C3 | DM2 sube fibra/leguminosas ↔ ERC restringe potasio/fósforo | Fibra de fuentes bajas en K/P | La nota del gap de fibra se orienta a vegetales/frutas bajos en K si `_has_renal` (`_FIBER_RENAL_NOTE`) | Árbitro `dm2+renal` en `build_condition_prompt` (modera leguminosas/granos) | `P2-RENAL-FIBER-NOTE`, árbitro dm2+renal |
| C4 | DASH/HTA empuja hoja verde ↔ warfarina exige vit-K CONSISTENTE (no máxima) | Consistencia, no maximización | El monitor `vitamin_k_consistency` (Guard 8d) se CONSUME: variabilidad alta → `_quality_degraded` reason=`vitamin_k_inconsistent` (banner, no retry) | `prompt_block` de la warfarina: "la consistencia TIENE PRECEDENCIA; no la interpretes como más hoja verde" | `P2-WARFARIN-VITK-CONSUME`, `P1-WARFARIN-VITAMIN-K` |

## Invariantes

- **I-CONF-1**: ningún surface determinista (panel/target) empuja un micronutriente en una dirección que un fármaco co-presente contraindica. El panel de micros recibe la señal de fármaco vía `k_elevating_med` (C1); las demás señales medicamentosas son advisory + gate FS9.
- **I-CONF-2**: todo conflicto fármaco↔condición levanta el gate FS9 (`requires_medication_review`) → el balance fino lo define el profesional. El árbitro determinista REDUCE el sesgo, no reemplaza la supervisión médica.
- **I-CONF-3**: la resolución por PROMPT es defensa-en-profundidad (LLM-trustable), NUNCA la única capa para un conflicto que un surface determinista puede empujar (ese se cierra en el target, como C1).

## Cómo verificar

```bash
pytest backend/tests/test_p2_audit_batch_19b.py -k conflict -v
```

El test ancla parsea esta tabla y exige que cada marker citado exista en el código fuente (si renombras un marker sin actualizar la tabla, el test falla).
