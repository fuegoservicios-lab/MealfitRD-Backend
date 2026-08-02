"""[P1-MICRO-REPORT-REFRESH · 2026-08-02] "1 de 3 días" en el banner, sodio 2266 mg — y el plan
real (8aa98e9c, forense 2026-08-02) recomputaba a Día1=1907/Día2=776 mg, NADIE sobre el techo
2000. `micronutrient_report` se persistió en la generación original y ninguna mutación posterior
lo refrescaba en las 4 superficies: `/swap-meal/persist` y `/regenerate-day` YA lo hacían inline
(P2-SWAP-MICROS-STALE · 2026-06-24 / P1-UPDATE-MICROS · 2026-06-23), `/fix-sodium-day` lo hereda
porque reusa `/swap-meal/persist` IN-PROCESS (P1-FIX-SODIUM-DAY · 2026-08-02) — el gap real era
`/shift-plan` (P0-1/api_shift_plan): el rolling window PODA `shifted_days` (días ya transcurridos,
archivados en `_archived_days`) sin recomputar nada, dejando `per_day_ceilings`/`per_day_floors`
evaluando un día que ya NO existe en la ventana visible. Un banner `micro_worst_day_ceiling`
marcado por ese día sobrevivía indefinidamente — el banner amarillo sobrevivía a su propia
solución (el día que lo causó ya se había ido).

Reconocimiento (forense SQL contra Neon, plan 8aa98e9c-b45e-4c92-9821-2a0a63704a7a):
    total_days_requested=30, days_len=2, archived_len=1, shift_days_accumulated=1,
    generation_status=partial. El caso `days=2` es NORMAL DEL ROLLING (no corrupción): un plan
    de 30 días con 1 día ya consumido/archivado + generación en curso del resto — verificado
    contra el mecanismo de `api_shift_plan` (`shifted_days[shift_amount:]` + `_archived_days`),
    NO reportado como bug. `micronutrient_report` YA estaba fresco en el momento del snapshot
    (`per_day_ceilings.days_above=0`, `_quality_degraded=None`) — probablemente refrescado por
    OTRA superficie (el chunk worker, `cron_tasks.py::_mn_recompute_chunk`) entre el forense
    original del owner y esta verificación; el gap estructural de `/shift-plan` (cero llamada a
    `recompute_micronutrient_report_for_plan` en todo el cuerpo de `api_shift_plan`) es real y
    reproducible independientemente de ese plan puntual — es lo que este archivo cierra.

Fix: helper SSOT `graph_orchestrator._refresh_micronutrient_report(plan, form_data, db=None,
*, surface=...)` — compone `recompute_micronutrient_report_for_plan` (reconstruye el reporte,
auto-limpia `micro_worst_day_ceiling` si se resolvió) + `apply_update_condition_ceilings`
(clear/mark bidireccional de los 4 `_PANEL_DEGRADED_REASONS`) + una re-evaluación bidireccional
EXTENDIDA (espejo EXACTO del bloque P2-REGEN-DAY-PANEL-REEVAL que `/regenerate-day` ya corre)
que además cubre `micro_worst_day`/`micro_worst_day_ceiling` — el paso 1 solo limpia el caso
ceiling, esta re-evaluación también limpia el caso floor y vuelve a MARCAR cualquiera de los dos
si el plan mutado los rompe de nuevo. Cableado a `/shift-plan`, gateado a `needs_shift` (contenido
de `days` realmente podado — el refill-only vía chunks async ya lo cubre el chunk worker) y al
mismo knob de rollback `MEALFIT_UPDATE_RECOMPUTE_MICROS` que las otras 3 superficies. El
health_profile se lee con el MISMO `cursor` que `api_shift_plan` ya tiene abierto bajo el `FOR
UPDATE` del plan (mismo patrón que las 2 lecturas de health_profile preexistentes en ese mismo
endpoint) — CERO reentrada a `connection_pool.connection()` mientras se sostiene el row lock
(P2-MUTATOR-PURITY, db_plans.py:562); el catálogo de nutrientes usa `db=None` (mismo lazy-load
TTL-cacheado que `/swap-meal/persist` y `/regenerate-day` ya aceptan — precedente establecido,
no una regresión nueva).

Delta de estimadores (documentado, NO arreglado aquí — scope aparte): el recompute descarta
"1 pizca de sal"/"pimienta al gusto" por falta de `density` en `master_ingredients`
(P1-MICRO-DENSITY-OBSERVABLE). Si el reporte original SÍ los contaba (heurística distinta o
`density` disponible en el momento de generación), la magnitud del delta 2266→1907/776 mg
observado por el owner (~360 mg) es consistente con sal-a-pizca + los platos efectivamente
cambiados entre la generación y el forense — la alineación de los DOS estimadores (generación vs
recompute) queda fuera de este fix.

Tooltip-anchor: P1-MICRO-REPORT-REFRESH
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_PLANS_SRC_PATH = _BACKEND / "routers" / "plans.py"
_APP_SRC_PATH = _BACKEND / "app.py"
_ORCH_SRC_PATH = _BACKEND / "graph_orchestrator.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extract_function_body(src: str, fn_name: str) -> str:
    pattern = re.compile(rf"def\s+{re.escape(fn_name)}\s*\(")
    m = pattern.search(src)
    assert m, f"`def {fn_name}(` no encontrado"
    start = m.start()
    next_def = re.search(r"\n(?:@router\.|@app\.|def\s)", src[start + 1:])
    end = (start + 1 + next_def.start()) if next_def else len(src)
    return src[start:end]


@pytest.fixture(scope="module")
def plans_src() -> str:
    return _read(_PLANS_SRC_PATH)


@pytest.fixture(scope="module")
def orch_src() -> str:
    return _read(_ORCH_SRC_PATH)


@pytest.fixture(scope="module")
def shift_plan_body(plans_src: str) -> str:
    return _extract_function_body(plans_src, "api_shift_plan")


@pytest.fixture(scope="module")
def refresh_helper_body(orch_src: str) -> str:
    return _extract_function_body(orch_src, "_refresh_micronutrient_report")


# ---------------------------------------------------------------------------
# 0. Marker anchor
# ---------------------------------------------------------------------------
def test_marker_anchored(plans_src: str, orch_src: str):
    assert plans_src.count("P1-MICRO-REPORT-REFRESH") >= 1
    assert orch_src.count("P1-MICRO-REPORT-REFRESH") >= 1


def test_marker_slug_matches_filename():
    expected_slug = "p1_micro_report_refresh"
    assert expected_slug in __file__.replace("\\", "/").lower()


# ---------------------------------------------------------------------------
# 1. El helper SSOT existe y compone los 3 pasos documentados
# ---------------------------------------------------------------------------
def test_helper_exists_and_composes_the_three_steps(refresh_helper_body: str):
    assert "recompute_micronutrient_report_for_plan(" in refresh_helper_body
    assert "apply_update_condition_ceilings(" in refresh_helper_body
    assert "_maybe_mark_panel_degraded(" in refresh_helper_body
    # orden: recompute PRIMERO (el resto opera sobre el panel YA fresco)
    i_recompute = refresh_helper_body.index("recompute_micronutrient_report_for_plan(")
    i_ceilings = refresh_helper_body.index("apply_update_condition_ceilings(")
    i_mark = refresh_helper_body.index("_maybe_mark_panel_degraded(")
    assert i_recompute < i_ceilings < i_mark


def test_helper_is_fail_safe(refresh_helper_body: str):
    assert refresh_helper_body.count("except Exception") >= 2, (
        "el panel es advisory: un fallo del paso 2/3 no puede tirar abajo al caller"
    )


# ---------------------------------------------------------------------------
# 2. Comportamiento funcional del helper (mock de build_micronutrient_report,
#    mismo patrón que test_p1_micro_degraded_stale_clear.py)
# ---------------------------------------------------------------------------
def _plan(reason=None, **extra):
    p = {
        "days": [{"day": 1, "meals": [{"meal": "Almuerzo", "name": "Moro",
                                       "ingredients": ["100 g de arroz"],
                                       "protein": 20, "carbs": 40, "fats": 10, "cals": 330}]}],
    }
    if reason:
        p["_quality_degraded"] = True
        p["_quality_degraded_reason"] = reason
        p["_quality_degraded_severity"] = "minor"
    p.update(extra)
    return p


def _run_refresh(plan, report, form_data=None, surface="test"):
    import graph_orchestrator as go
    import micronutrients as mn

    _orig = mn.build_micronutrient_report
    mn.build_micronutrient_report = lambda *a, **k: report
    try:
        return go._refresh_micronutrient_report(plan, form_data or {}, db=None, surface=surface)
    finally:
        mn.build_micronutrient_report = _orig


def test_clears_ceiling_reason_when_resolved():
    """(mirror de P1-MICRO-DEGRADED-STALE-CLEAR, verificado a través del NUEVO helper)."""
    plan = _plan(reason="micro_worst_day_ceiling:sodium")
    ok = _run_refresh(plan, {
        "coverage": 0.9, "gaps": [],
        "per_day_ceilings": {"flagged": False, "days_above": 0, "days_evaluated": 2,
                              "worst_day": {"high": [], "day_index": 0}},
        "per_day_floors": {"flagged": False, "days_below": 0},
    })
    assert ok is True
    assert "_quality_degraded" not in plan and "_quality_degraded_reason" not in plan


def test_clears_floor_reason_micro_worst_day_when_resolved():
    """Diferenciador del helper NUEVO: `recompute_micronutrient_report_for_plan` (P1-MICRO-
    DEGRADED-STALE-CLEAR) solo auto-limpia el caso CEILING — el caso FLOOR (`micro_worst_day`,
    per_day_floors) quedaba sin cubrir. La re-evaluación bidireccional extendida (paso 3, espejo
    de P2-REGEN-DAY-PANEL-REEVAL) sí lo cubre."""
    plan = _plan(reason="micro_worst_day")
    ok = _run_refresh(plan, {
        "coverage": 0.9, "gaps": [],
        "per_day_ceilings": {"flagged": False, "days_above": 0},
        "per_day_floors": {"flagged": False, "days_below": 0, "worst_day": {"low": [], "day_index": 0}},
    })
    assert ok is True
    assert "_quality_degraded" not in plan, (
        "el helper debe limpiar micro_worst_day (floor) igual que ya limpia el ceiling — "
        "sin la re-evaluación extendida este caso quedaba stale para siempre"
    )


def test_marks_new_ceiling_violation_when_none_existed():
    plan = _plan(reason=None)
    ok = _run_refresh(plan, {
        "coverage": 0.9, "gaps": [],
        "per_day_ceilings": {"flagged": True, "days_above": 1,
                              "worst_day": {"high": ["sodium_mg"], "day_index": 0}},
        "per_day_floors": {"flagged": False, "days_below": 0},
    })
    assert ok is True
    assert plan.get("_quality_degraded") is True
    assert plan.get("_quality_degraded_reason") == "micro_worst_day_ceiling"


def test_kept_when_ceiling_still_violated():
    plan = _plan(reason="micro_worst_day_ceiling:sodium")
    _run_refresh(plan, {
        "coverage": 0.9, "gaps": [],
        "per_day_ceilings": {"flagged": True, "days_above": 1,
                              "worst_day": {"high": ["sodium_mg"], "day_index": 0}},
        "per_day_floors": {"flagged": False, "days_below": 0},
    })
    assert plan.get("_quality_degraded") is True, "techo aún violado → el banner NO miente"


def test_non_panel_reasons_untouched():
    """Razones clínicas/banda/fallback (review_failed, low_band_*, max_attempts) jamás se tocan,
    incluso si el panel recomputado muestra violaciones."""
    plan = _plan(reason="review_failed")
    _run_refresh(plan, {
        "coverage": 0.9, "gaps": [],
        "per_day_ceilings": {"flagged": True, "days_above": 1,
                              "worst_day": {"high": ["sodium_mg"], "day_index": 0}},
        "per_day_floors": {"flagged": False, "days_below": 0},
    })
    assert plan.get("_quality_degraded_reason") == "review_failed", (
        "razones no-panel jamás se tocan, aunque el panel muestre un techo roto"
    )


def test_returns_false_on_garbage_input():
    import graph_orchestrator as go
    assert go._refresh_micronutrient_report(None, {}) is False
    assert go._refresh_micronutrient_report({}, {}) in (True, False)


# ---------------------------------------------------------------------------
# 3. Cableado en /shift-plan: gateado a needs_shift + mismo knob de rollback
# ---------------------------------------------------------------------------
def test_shift_plan_calls_the_refresh_helper(shift_plan_body: str):
    assert "_refresh_micronutrient_report" in shift_plan_body, (
        "el shift debe recomputar el panel de micros tras podar días — sin esto el banner "
        "sobrevive al día que lo causó"
    )


def test_shift_plan_refresh_gated_on_needs_shift(shift_plan_body: str):
    i_marker = shift_plan_body.index("[P1-MICRO-REPORT-REFRESH")
    i_call = shift_plan_body.index("_refresh_micronutrient_report")
    # el `if needs_shift` que gatea el bloque debe preceder la llamada, sin una función
    # nueva metida en medio (mismo patrón de ventana que el resto de la suite).
    window = shift_plan_body[i_marker:i_call]
    assert "if needs_shift" in window, (
        "el refresh debe gatearse a needs_shift (contenido de days REALMENTE podado) — el "
        "refill-only (solo encola chunks async) no necesita este refresh síncrono"
    )
    assert "\ndef " not in window.split("if needs_shift")[-1]


def test_shift_plan_refresh_respects_rollback_knob(shift_plan_body: str):
    i_marker = shift_plan_body.index("[P1-MICRO-REPORT-REFRESH")
    i_call = shift_plan_body.index("_refresh_micronutrient_report")
    window = shift_plan_body[i_marker:i_call]
    assert "MEALFIT_UPDATE_RECOMPUTE_MICROS" in window, (
        "mismo knob de rollback que swap-persist/regen-day/chat-modify — un incidente debe "
        "poder apagar las 4 superficies de una vez"
    )


def test_shift_plan_refresh_is_best_effort(shift_plan_body: str):
    i = shift_plan_body.index("_refresh_micronutrient_report")
    tail = shift_plan_body[i:i + 400]
    assert "except Exception" in tail, "el refresh nunca puede tirar abajo el shift"


# ---------------------------------------------------------------------------
# 4. Puridad (P2-MUTATOR-PURITY): el bloque del shift no reentra al pool
# ---------------------------------------------------------------------------
def test_shift_plan_refresh_block_does_not_reenter_pool(shift_plan_body: str):
    """El bloque que rodea la llamada al helper NO debe abrir una conexión nueva del pool
    (`with connection_pool.connection()`), ni usar `execute_sql_query`/`get_user_profile`/
    `IngredientNutritionDB(` — esas son exactamente las 3 formas de reentrar al pool mientras
    el `FOR UPDATE` del plan sigue sostenido (db_plans.py:562, P2-MUTATOR-PURITY). El único I/O
    permitido dentro del bloque es `cursor.execute`/`cursor.fetchone` sobre el cursor YA abierto
    (mismo patrón que las 2 lecturas de health_profile preexistentes en este mismo endpoint)."""
    i_marker = shift_plan_body.index("[P1-MICRO-REPORT-REFRESH")
    i_call = shift_plan_body.index("_refresh_micronutrient_report")
    i_end = shift_plan_body.index("\n\n", i_call)
    block = shift_plan_body[i_marker:i_end]
    # Strip comment lines (`# ...`) antes de escanear: el propio comentario del fix EXPLICA
    # por qué no se reentra al pool citando la API prohibida en prosa — sin este strip el test
    # fallaría contra su propia documentación (no contra código ejecutable).
    block_no_comments = re.sub(r"#[^\n]*", "", block)
    for forbidden in ("connection_pool.connection(", "execute_sql_query(",
                       "get_user_profile(", "IngredientNutritionDB("):
        assert forbidden not in block_no_comments, (
            f"el bloque del refresh en /shift-plan usa `{forbidden}` — eso reentra al pool "
            f"mientras el FOR UPDATE del plan está sostenido (riesgo de starvation bajo carga "
            f"concurrente, db_plans.py:562)"
        )
    assert "cursor.execute(" in block_no_comments, "debe leer health_profile con el cursor YA abierto"


# ---------------------------------------------------------------------------
# 5. (a)/(b) Funcional E2E: /swap-meal/persist recomputa y clear/marca el banner
# ---------------------------------------------------------------------------
def _swap_persist_env(monkeypatch, plan_data):
    """Cablea los mismos boundaries que el resto de la suite de swap
    (db_core.execute_sql_query, db.get_user_profile) + `db_plans.update_plan_data_atomic`
    parcheado para invocar el mutator REAL sobre `plan_data` (in-memory, sin Postgres) — deja
    correr la cadena de producción completa: `_swap_mutator` → `recompute_micronutrient_report_
    for_plan` → `apply_update_condition_ceilings` → `_maybe_mark_panel_degraded`."""
    import copy as _copy
    import db_core
    import db
    import db_plans

    def _fake_execute_sql_query(query, params=None, fetch_one=False, fetch_all=False):
        # Distingue el SELECT de ownership (`SELECT id FROM meal_plans...`, usado por
        # api_swap_meal_persist) del SELECT de contenido (`SELECT plan_data FROM
        # meal_plans...`, usado por fix-sodium-day tanto en la lectura inicial como en el
        # re-read post-persist) — mismo criterio que el resto de la suite de swap.
        if "SELECT plan_data" in query:
            return {"plan_data": _copy.deepcopy(plan_data)}
        return {"id": "plan-refresh-1"}

    monkeypatch.setattr(db_core, "execute_sql_query", _fake_execute_sql_query)
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {
        "health_profile": {"gender": "female", "age": 30, "allergies": [], "dietType": None}
    })

    def _fake_atomic(plan_id, mutator, lock_timeout_ms=None, *, user_id=None):
        result = mutator(plan_data)
        return result if isinstance(result, dict) else plan_data

    monkeypatch.setattr(db_plans, "update_plan_data_atomic", _fake_atomic)


def _mock_report_by_meal_name(low_sodium_name: str):
    """Reporte sintético: si `low_sodium_name` aparece en CUALQUIER meal del plan (búsqueda
    cross-day — el swap puede tocar cualquier día, no solo el 0), el techo NO está roto;
    de lo contrario, el techo SÍ está roto por sodio. Evita depender del catálogo real /
    Postgres (mismo criterio que el estimador sintético de test_p1_fix_sodium_day.py)."""
    def _report(plan, db, **kw):
        names = [m.get("name") for d in (plan.get("days") or []) if isinstance(d, dict)
                 for m in (d.get("meals") or []) if isinstance(m, dict)]
        if low_sodium_name in names:
            return {
                "coverage": 0.9, "gaps": [],
                "per_day_ceilings": {"flagged": False, "days_above": 0,
                                      "worst_day": {"high": [], "day_index": 0}},
                "per_day_floors": {"flagged": False, "days_below": 0},
            }
        return {
            "coverage": 0.9, "gaps": [],
            "per_day_ceilings": {"flagged": True, "days_above": 1,
                                  "worst_day": {"high": ["sodium_mg"], "day_index": 0}},
            "per_day_floors": {"flagged": False, "days_below": 0},
        }
    return _report


def test_a_swap_persist_low_sodium_clears_the_reason(monkeypatch):
    import micronutrients as mn
    import routers.plans as _rp

    plan_data = {
        "days": [{"day": 1, "meals": [{"name": "Sancocho Alto Sodio", "meal": "Almuerzo",
                                        "ingredients": ["300g Carne de res"],
                                        "ingredients_raw": ["300g Carne de res"],
                                        "protein": 30, "carbs": 45, "fats": 18, "cals": 500}]}],
        "_quality_degraded": True,
        "_quality_degraded_reason": "micro_worst_day_ceiling:sodium",
        "_quality_degraded_severity": "minor",
    }
    _swap_persist_env(monkeypatch, plan_data)
    monkeypatch.setattr(mn, "build_micronutrient_report",
                         _mock_report_by_meal_name("Pollo Bajo Sodio"))

    resp = _rp.api_swap_meal_persist(
        "plan-refresh-1",
        data={
            "day_index": 0, "meal_index": 0,
            "new_meal": {"name": "Pollo Bajo Sodio", "meal": "Almuerzo",
                         "ingredients": ["150g Pollo"], "ingredients_raw": ["150g Pollo"],
                         "cals": 400, "protein": 35, "carbs": 20, "fats": 10},
        },
        verified_user_id="user-1",
    )

    assert resp["success"] is True
    report = plan_data.get("micronutrient_report")
    assert report is not None, "el swap debe recomputar y persistir micronutrient_report"
    assert report["per_day_ceilings"]["flagged"] is False
    assert "sodium_mg" not in (report["per_day_ceilings"]["worst_day"].get("high") or [])
    assert "_quality_degraded" not in plan_data, (
        "el banner micro_worst_day_ceiling debe LIMPIARSE tras el swap a un plato bajo en sodio"
    )


def test_b_swap_persist_to_worse_meal_marks_the_reason(monkeypatch):
    import micronutrients as mn
    import routers.plans as _rp

    plan_data = {
        "days": [{"day": 1, "meals": [{"name": "Pollo Bajo Sodio", "meal": "Almuerzo",
                                        "ingredients": ["150g Pollo"],
                                        "ingredients_raw": ["150g Pollo"],
                                        "protein": 35, "carbs": 20, "fats": 10, "cals": 400}]}],
    }
    _swap_persist_env(monkeypatch, plan_data)
    monkeypatch.setattr(mn, "build_micronutrient_report",
                         _mock_report_by_meal_name("Pollo Bajo Sodio"))

    resp = _rp.api_swap_meal_persist(
        "plan-refresh-1",
        data={
            "day_index": 0, "meal_index": 0,
            "new_meal": {"name": "Sancocho Alto Sodio", "meal": "Almuerzo",
                         "ingredients": ["300g Carne de res"],
                         "ingredients_raw": ["300g Carne de res"],
                         "cals": 500, "protein": 30, "carbs": 45, "fats": 18},
        },
        verified_user_id="user-1",
    )

    assert resp["success"] is True
    report = plan_data.get("micronutrient_report")
    assert report is not None
    assert report["per_day_ceilings"]["flagged"] is True
    assert "sodium_mg" in (report["per_day_ceilings"]["worst_day"].get("high") or [])
    assert plan_data.get("_quality_degraded") is True, (
        "el swap a un plato alto en sodio debe MARCAR el banner (bidireccional: el helper "
        "también marca, no solo limpia)"
    )
    assert plan_data.get("_quality_degraded_reason") == "micro_worst_day_ceiling"


# ---------------------------------------------------------------------------
# 6. (d) fix-sodium-day end-to-end: panel coherente con la respuesta
# ---------------------------------------------------------------------------
def test_d_fix_sodium_day_end_to_end_panel_coherent_with_response(monkeypatch):
    """A diferencia de `test_p1_fix_sodium_day.py` (que mockea `api_swap_meal_persist`
    directamente), aquí se deja correr la función REAL — la cadena completa fix-sodium-day →
    swap_meal (mock del LLM) → api_swap_meal_persist REAL → recompute — para verificar que el
    panel persistido queda coherente con lo que la respuesta le informa al usuario."""
    import micronutrients as mn
    import routers.plans as _rp
    import graph_orchestrator as go

    plan_data = {
        "days": [
            {"meals": [{"name": "Avena", "meal": "Desayuno", "ingredients": ["100g Avena"],
                        "ingredients_raw": ["100g Avena"],
                        "cals": 300, "protein": 10, "carbs": 40, "fats": 8}]},
            {"meals": [{"name": "Ricotta con Camarones", "meal": "Cena",
                        "ingredients": ["150g Camarones", "100g Ricotta"],
                        "ingredients_raw": ["150g Camarones", "100g Ricotta"],
                        "cals": 480, "protein": 32, "carbs": 15, "fats": 22}]},
        ],
        "micronutrient_report": {
            "per_day_ceilings": {"flagged": True, "days_above": 1,
                                  "worst_day": {"high": ["sodium_mg"], "day_index": 1}},
        },
        "_quality_degraded": True,
        "_quality_degraded_reason": "micro_worst_day_ceiling:sodium",
        "_quality_degraded_severity": "minor",
    }
    _swap_persist_env(monkeypatch, plan_data)
    monkeypatch.setattr(mn, "build_micronutrient_report",
                         _mock_report_by_meal_name("Camarones al Ajillo Ligero"))
    monkeypatch.setattr(go, "_meal_sodium_mg",
                         lambda meal, _db: 1100.0 if "Camarones" in str(meal.get("ingredients_raw")) and
                         "Ligero" not in str(meal.get("name")) else (300.0 if "Camarones" in
                         str(meal.get("ingredients_raw")) else 50.0))
    monkeypatch.setattr(go, "_sodium_day_ceiling_mg_for_banner", lambda form_data=None: 1000.0)

    def _fake_swap_meal(meal_form):
        return {
            "name": "Camarones al Ajillo Ligero", "desc": "Camarones salteados, bajo en sodio.",
            "cals": 420, "prep_time": 20,
            "recipe": ["Saltea los camarones con ajo y limón."],
            "ingredients": ["150g Camarones frescos", "1 cda Aceite de oliva"],
            "ingredients_raw": ["150g Camarones frescos", "1 cda Aceite de oliva"],
        }

    monkeypatch.setattr(_rp, "swap_meal_with_consent", _fake_swap_meal)
    monkeypatch.setattr(_rp, "log_api_usage", lambda *a, **k: None)

    result = _rp.api_fix_sodium_day("plan-refresh-1", data={}, verified_user_id="user-1", _rl=None)

    assert result["fixed"] is True
    report = plan_data.get("micronutrient_report")
    assert report is not None
    assert report["per_day_ceilings"]["flagged"] is False, (
        "el panel persistido tras fix-sodium-day debe reflejar el plato nuevo, no el viejo"
    )
    assert "_quality_degraded" not in plan_data, (
        "fix-sodium-day hereda el recompute de swap-persist (in-process) — el banner debe "
        "quedar coherente con la respuesta 'fixed: true' que ve el usuario"
    )
