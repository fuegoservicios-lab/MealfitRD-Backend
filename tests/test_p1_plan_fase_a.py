"""[P1-PLAN-FASE-A · 2026-09-11] Fase A del plan de pendientes (`docs/plan_pendientes_2026_09_11.md`):
lo que la verificación de la auditoría de arquitectura dejó abierto y no necesitaba una decisión del dueño.

  A6  `_SLOT_KEY_MAP`/`_canonical_slot_fractions` → `nutrition_calculator` (junto a `MEAL_SLOT_SPLITS`);
      `_line_sodium_mg`/`_meal_sodium_mg` → `nutrition_db` (junto al primitivo). El god-file los re-exporta.
  A3  «sin dato» de sodio deja de ser 0: `line_sodium_mg_or_none`, `meal_sodium_detail`, y el autofix del día
      anota `_sodium_unknown_lines`.
  A4  `review_plan_node` consume `_shopping_coherence_unevaluable`: history `guard_unevaluable`, marca
      `_review_unevaluable_checks`, alerta en modo `block`. Un guard que no evaluó no es un guard que aprobó.
  A5  El coste de la fila entra en la huella de la lista: un recálculo tras cambiar precios re-encola la
      proyección aunque no se mueva ni una cantidad.
  A7  `apply_library_recipe` escala el agua medida de la receta congelada con el factor implícito (gramos
      servidos / gramos de la plantilla) y declara `_recipe_water_unscaled` cuando no puede.
  A8  Tres docs dejan de reabrir gaps cerrados y la tabla de surfaces pierde las referencias de línea muertas.

Cada test expresa el comportamiento ESPERADO. Ninguno codifica el defecto como especificación.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# A6 · extracción del god-file
# ---------------------------------------------------------------------------

def test_a6_reparto_canonico_vive_en_nutrition_calculator_y_el_grafo_lo_reexporta():
    import graph_orchestrator as go
    import nutrition_calculator as nc
    assert go._canonical_slot_fractions is nc._canonical_slot_fractions
    assert go._SLOT_KEY_MAP is nc._SLOT_KEY_MAP
    assert nc.canonical_slot_fractions is nc._canonical_slot_fractions
    go_src = _src("graph_orchestrator.py")
    assert "def _canonical_slot_fractions(" not in go_src
    assert "\n_SLOT_KEY_MAP = {" not in go_src
    # misma regla, mismos datos: el reparto de 4 comidas es el fisiológico, no el plano
    fr = nc._canonical_slot_fractions([{"meal": m} for m in ("Desayuno", "Almuerzo", "Merienda", "Cena")])
    assert fr == pytest.approx([0.20, 0.35, 0.15, 0.30])
    assert nc._canonical_slot_fractions([]) == []


def test_a6_deterministic_day_ya_no_pide_franjas_al_grafo():
    import deterministic_day as dd
    src = _src("deterministic_day.py")
    assert "from graph_orchestrator import _SLOT_KEY_MAP" not in src
    assert "from graph_orchestrator import _canonical_slot_fractions" not in src
    assert "from nutrition_calculator import _SLOT_KEY_MAP" in src
    assert dd._etiqueta_a_franja("Merienda Nocturna") == "merienda"
    assert dd._fracciones_por_franja(["Desayuno", "Almuerzo", "Cena"]) == pytest.approx([0.30, 0.40, 0.30])


def test_a6_estimador_de_sodio_vive_en_nutrition_db_y_el_grafo_lo_reexporta():
    import graph_orchestrator as go
    import nutrition_db as nd
    assert go._line_sodium_mg is nd._line_sodium_mg
    assert go._meal_sodium_mg is nd._meal_sodium_mg
    assert go.meal_sodium_detail is nd.meal_sodium_detail
    go_src = _src("graph_orchestrator.py")
    assert "def _line_sodium_mg(" not in go_src and "def _meal_sodium_mg(" not in go_src
    # el autofix sigue delegando en los nombres de módulo (los tests que parchean `go.` siguen valiendo)
    i = go_src.find("def _day_sodium_autofix(")
    assert "return _line_sodium_mg(_s, db)" in go_src[i:i + 5000]
    assert "return sum(_meal_sodium_mg(_m, db)" in go_src[i:i + 5000]


def test_a6_el_god_file_quedo_bajo_su_tope():
    n = len(_src("graph_orchestrator.py").splitlines())
    assert n < 53100, f"{n} líneas: la extracción tenía que dejar aire, no consumirlo"


# ---------------------------------------------------------------------------
# A3 · «sin dato» no es cero
# ---------------------------------------------------------------------------

class _DB:
    """Catálogo de mentira: `None` = línea que no resuelve; `{"sodium_mg": None}` = fila sin dato."""

    def micros_from_ingredient_string(self, s):
        low = str(s).lower()
        if "pollo" in low:
            return {"sodium_mg": 120.0, "grams": 150.0}
        if "sal" in low:
            return {"sodium_mg": 2000.0, "grams": 5.0}
        if "sindato" in low:
            return {"sodium_mg": None, "grams": 80.0}
        return None


def test_a3_line_sodium_or_none_distingue_cero_de_sin_dato():
    from nutrition_db import line_sodium_mg_or_none, _line_sodium_mg
    db = _DB()
    assert line_sodium_mg_or_none("150 g de Pollo", db) == 120.0
    assert line_sodium_mg_or_none("80 g de Sindato", db) is None
    assert line_sodium_mg_or_none("1 misterio", db) is None
    # el contrato histórico se conserva para los llamadores que suman
    assert _line_sodium_mg("150 g de Pollo", db) == 120.0
    assert _line_sodium_mg("1 misterio", db) == 0.0


def test_a3_meal_sodium_detail_cuenta_las_lineas_sin_dato():
    from nutrition_db import meal_sodium_detail, _meal_sodium_mg
    db = _DB()
    meal = {"ingredients": ["150 g de Pollo", "80 g de Sindato", "1 misterio", "5 g de Sal"]}
    mg, unknown = meal_sodium_detail(meal, db)
    assert mg == pytest.approx(2120.0)
    assert unknown == 2
    assert _meal_sodium_mg(meal, db) == pytest.approx(2120.0)
    assert meal_sodium_detail("no es dict", db) == (0.0, 0)
    # ingredients_raw manda cuando existe (mismo criterio que el panel)
    meal2 = {"ingredients": ["1 misterio"], "ingredients_raw": ["150 g de Pollo"]}
    assert meal_sodium_detail(meal2, db) == (120.0, 0)


def test_a3_el_autofix_del_dia_anota_las_lineas_sin_dato(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "SODIUM_DAY_AUTOFIX_ENABLED", True)
    db = _DB()
    dia_dudoso = {"day": 1, "meals": [{"name": "A", "ingredients": ["150 g de Pollo", "1 misterio", "80 g de Sindato"]}]}
    dia_limpio = {"day": 2, "meals": [{"name": "B", "ingredients": ["150 g de Pollo"]}]}
    go._day_sodium_autofix([dia_dudoso, dia_limpio], form_data={}, db=db)
    assert dia_dudoso["_sodium_unknown_lines"] == 2
    assert "_sodium_unknown_lines" not in dia_limpio


# ---------------------------------------------------------------------------
# A4 · el revisor consume el guard no evaluable
# ---------------------------------------------------------------------------

def _bypass_form_data():
    return {"user_id": "guest", "allergies": [], "medicalConditions": [], "dislikes": [],
            "dietType": "balanced", "_days_to_generate": 3}


def _minimal_plan():
    return {
        "calories": 2000,
        "macros": {"protein": 150, "carbs": 200, "fats": 67},
        "days": [{"day": 1, "meals": [
            {"meal": "almuerzo", "name": "Pollo con arroz", "ingredients": ["200 g pollo", "150 g arroz"],
             "recipe": ["Mise en place: pesa el pollo y lava el arroz.",
                        "El Toque de Fuego: cocina el pollo 8-10 min a fuego medio y hierve el arroz 15 min.",
                        "Montaje: sirve el pollo sobre el arroz."],
             "protein": 150, "carbs": 200, "fats": 67, "cals": 2000}]}],
    }


def _state(plan):
    return {"plan_result": plan, "form_data": _bypass_form_data(), "taste_profile": "", "attempt": 1,
            "rejection_reasons": [], "_rejection_severity": "minor", "request_id": "test-fase-a"}


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_a4_guard_no_evaluable_deja_history_marca_y_alerta_en_block(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "block")
    emitidas = []
    monkeypatch.setattr(go, "execute_sql_write", lambda sql, params=None, *a, **k: emitidas.append((sql, params)))
    plan = _minimal_plan()
    plan["_shopping_coherence_unevaluable"] = {"stage": "expected_sum_from_recipes", "error": "KeyError",
                                               "detail": "'x'", "at": "2026-09-11T00:00:00+00:00"}
    result = _run(go.review_plan_node(_state(plan)))
    hist = plan["_shopping_coherence_block_history"]
    assert hist[-1]["action_taken"] == "guard_unevaluable"
    assert hist[-1]["block_set"] is False
    assert hist[-1]["unevaluable_stage"] == "expected_sum_from_recipes"
    assert plan["_review_unevaluable_checks"] == ["shopping_coherence"]
    # no se rechaza: reintentar repetiría el mismo cómputo determinista con el mismo fallo
    assert "COHERENCIA RECETAS LISTA" not in " ".join(result.get("rejection_reasons") or [])
    # pero el operador se entera
    assert len(emitidas) == 1
    sql, params = emitidas[0]
    assert "shopping_coherence_guard_unevaluable" in sql and "'warning'" in sql
    assert params[0] == "shopping_coherence_guard_unevaluable:guest:no_plan_id"
    assert json.loads(params[3])["stage"] == "expected_sum_from_recipes"


def test_a4_en_warn_hay_constancia_pero_no_alerta(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setenv("MEALFIT_SHOPPING_COHERENCE_GUARD", "warn")
    emitidas = []
    monkeypatch.setattr(go, "execute_sql_write", lambda *a, **k: emitidas.append(a))
    plan = _minimal_plan()
    plan["_shopping_coherence_unevaluable"] = {"stage": "run_shopping_coherence_guard", "error": "TypeError"}
    _run(go.review_plan_node(_state(plan)))
    assert plan["_shopping_coherence_block_history"][-1]["action_taken"] == "guard_unevaluable"
    assert plan["_review_unevaluable_checks"] == ["shopping_coherence"]
    assert emitidas == []


def test_a4_sin_flag_el_revisor_no_inventa_nada(monkeypatch):
    import graph_orchestrator as go
    monkeypatch.setattr(go, "execute_sql_write", lambda *a, **k: pytest.fail("no debía emitir alerta"))
    plan = _minimal_plan()
    _run(go.review_plan_node(_state(plan)))
    assert "_review_unevaluable_checks" not in plan
    assert not any(h.get("action_taken") == "guard_unevaluable"
                   for h in (plan.get("_shopping_coherence_block_history") or []))


def test_a4_la_alerta_esta_documentada_y_el_valor_es_canonico():
    tabla = _src("docs/system_alerts_resolution_table.md")
    assert "`shopping_coherence_guard_unevaluable:<user_id>:<plan_id>`" in tabla
    surfaces = _src("docs/coherence_surfaces_table.md")
    assert "`guard_unevaluable`" in surfaces and "_shopping_coherence_unevaluable" in surfaces
    go_src = _src("graph_orchestrator.py")
    assert 'alert_key = f"shopping_coherence_guard_unevaluable:{user_id}:{plan_id}"' in go_src
    i = go_src.find('coherence_block = plan.get("_shopping_coherence_block") or []')
    j = go_src.find('skeleton_fidelity_errors = plan.get("_skeleton_fidelity_errors", [])')
    assert 0 < i < j
    assert '"guard_unevaluable"' in go_src[i:j] and "_emit_shopping_guard_unevaluable_alert(state, _coh_unev)" in go_src[i:j]


# ---------------------------------------------------------------------------
# A5 · el precio entra en la huella
# ---------------------------------------------------------------------------

def _pd(cost=100.0, qty=907):
    return {
        "calc_household_multiplier": 1.5, "total_days_requested": 30, "days": [{}] * 3, "_archived_days": [],
        "aggregated_shopping_list_weekly": [
            {"name": "Pollo", "base_qty": qty, "base_unit": "g", "market_qty": 2, "market_unit": "lb",
             "estimated_cost_rd": cost},
            {"name": "Arroz", "base_qty": 2000, "base_unit": "g", "market_qty": 1, "market_unit": "funda",
             "estimated_cost_rd": 80.0},
        ],
    }


def test_a5_un_cambio_de_precio_cambia_la_huella_sin_mover_cantidades():
    from shopping.projection.reprojection import shopping_list_fingerprint, _FINGERPRINT_ROW_KEYS
    assert "estimated_cost_rd" in _FINGERPRINT_ROW_KEYS
    assert shopping_list_fingerprint(_pd(cost=100.0)) != shopping_list_fingerprint(_pd(cost=150.0))
    assert shopping_list_fingerprint(_pd()) == shopping_list_fingerprint(_pd())
    # sin la clave la huella sigue siendo estable (filas viejas sin coste no se re-encolan solas)
    sin = _pd()
    for r in sin["aggregated_shopping_list_weekly"]:
        r.pop("estimated_cost_rd")
    assert shopping_list_fingerprint(sin) == shopping_list_fingerprint(json.loads(json.dumps(sin)))


# ---------------------------------------------------------------------------
# A7 · el agua de la receta congelada en el camino LLM
# ---------------------------------------------------------------------------

@pytest.fixture
def biblioteca(monkeypatch):
    import recipe_library as rl
    import dish_registry as dr
    monkeypatch.setenv("MEALFIT_RECIPE_LIBRARY_SELECT", "1")
    monkeypatch.setattr(rl, "library_select_enabled", lambda: True)
    monkeypatch.setattr(rl, "_name_index", lambda country="DO": {"arroz con pollo": "tpl_x"})
    monkeypatch.setattr(rl, "_library", lambda country="DO": {
        "tpl_x": {"pasos": ["Hierve 2 tazas de agua con una pizca de sal.", "Añade el arroz y el pollo."]}})
    monkeypatch.setattr(rl, "_foods_de_plantilla", lambda tid, country: frozenset({"arroz", "pollo"}))
    monkeypatch.setattr(dr, "templates_by_id", lambda country="DO": {
        "tpl_x": {"constituents": [{"name": "Arroz", "canonical": "Arroz", "grams": 100.0},
                                   {"name": "Pollo", "canonical": "Pollo", "grams": 100.0}]}})
    return rl


def test_a7_el_agua_escala_con_el_factor_implicito(biblioteca):
    rl = biblioteca
    meal = {"name": "Arroz con pollo", "ingredients": ["200 g de Arroz", "200 g de Pollo"], "recipe": ["x"]}
    assert rl.apply_library_recipe(meal) is True
    assert meal["_recipe_source"] == "library"
    assert meal["_recipe_scale_factor"] == pytest.approx(2.0)
    assert meal["_recipe_water_scaled"] is True
    assert meal["recipe"][0].startswith("Hierve 4 tazas de agua")
    assert meal["recipe"][1] == "Añade el arroz y el pollo."


def test_a7_sin_gramos_no_hay_factor_y_se_declara(biblioteca):
    rl = biblioteca
    meal = {"name": "Arroz con pollo", "ingredients": ["2 tazas de Arroz", "1 unidad de Pollo"], "recipe": ["x"]}
    assert rl.apply_library_recipe(meal) is True
    assert meal["_recipe_water_unscaled"] is True
    assert "_recipe_scale_factor" not in meal
    assert meal["recipe"][0] == "Hierve 2 tazas de agua con una pizca de sal."


def test_a7_una_racion_igual_a_la_base_no_toca_nada(biblioteca):
    rl = biblioteca
    meal = {"name": "Arroz con pollo", "ingredients": ["100 g de Arroz", "100 g de Pollo"], "recipe": ["x"]}
    assert rl.apply_library_recipe(meal) is True
    assert meal["_recipe_scale_factor"] == pytest.approx(1.0)
    assert "_recipe_water_scaled" not in meal and "_recipe_water_unscaled" not in meal
    assert meal["recipe"][0] == "Hierve 2 tazas de agua con una pizca de sal."


def test_a7_una_receta_sin_agua_medida_no_se_marca(biblioteca, monkeypatch):
    rl = biblioteca
    monkeypatch.setattr(rl, "_library", lambda country="DO": {"tpl_x": {"pasos": ["Saltea el arroz con el pollo."]}})
    meal = {"name": "Arroz con pollo", "ingredients": ["2 tazas de Arroz", "1 unidad de Pollo"], "recipe": ["x"]}
    assert rl.apply_library_recipe(meal) is True
    assert "_recipe_water_unscaled" not in meal and "_recipe_water_scaled" not in meal


# ---------------------------------------------------------------------------
# A8 · docs que dejaban de decir la verdad
# ---------------------------------------------------------------------------

def test_a8_docs_actualizados():
    f3 = _src("docs/plan_policy_f3.md")
    assert "ARQ27-P1-07" in f3 and "market_check_applied" in f3
    assert "nadie se lo pasa" not in f3
    f6 = _src("docs/dish_registry_f6.md")
    assert "prep_minutes_source" in f6 and "P1-MINUTOS-DE-LA-RECETA" in f6
    surfaces = _src("docs/coherence_surfaces_table.md")
    for muerta in ("graph_orchestrator.py:6185", "review_plan_node:7704", "graph_orchestrator.py:7860",
                   "cron_tasks.py:22678", "routers/plans.py:4006", "tools.py:514"):
        assert muerta not in surfaces, f"referencia de línea muerta: {muerta}"


def test_marker_bumpeado():
    """El marker avanza con cada lote; lo que queda anclado es que la Fase A se cerró con su comentario en
    `app.py` (el cross-link marker↔test lo hace `test_p2_hist_audit_14_marker_test_link`)."""
    import app
    assert "[P1-PLAN-FASE-A · 2026-09-11]" in _src("app.py")
    assert app._LAST_KNOWN_PFIX.startswith("P1-PLAN-") and "2026-09-11" in app._LAST_KNOWN_PFIX
