"""[P2-AUDIT-V7-BATCH · 2026-07-04] Test ancla de los 12 P2 del audit v7.

Cada sección ancla UN gap cerrado (source-anchor con tooltip del código de prod +
funcional donde el helper es invocable barato). Numeración = plan del audit v7:

  P2-1  recompute del panel de micros post-motor en form-gen (+ convergencia).
  P2-2  micro-recheck post-motor (closer re-fire idempotente + re-trim + re-quantize).
  P2-3  flip default de código MEALFIT_CARB_TO_PROTEIN_SWAP → True.
  P2-4  /recipe/expand con paridad de motor (engine → micros → band-parity).
  P2-5  gate suave de TRANSFORMACIÓN (creatividad; advisory en intento final).
  P2-6  coherencia cross-semana determinista en el merge de chunks + chip.
  P2-7  clamp timetemp en TODOS los pasos (pasivos marinar/reposar exentos).
  P2-8  lista de compras por pasillo en la UI viva (frontend).
  P2-9  CTA "limitado por Nevera" (frontend consume _quality_degraded_pantry_limited).
  P2-10 nombres del mismo slot en otros días → prompt de chat-modify.
  P2-11 contrato de sincronía regen-day inline ↔ helper SSOT del motor.
  P2-12 alert I5 per-plan en el path approved-con-residuo.
"""
import os
import re

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)
_FRONTEND = os.path.join(os.path.dirname(_BACKEND), "frontend")


def _read(*parts) -> str:
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


_GO = _read(_BACKEND, "graph_orchestrator.py")
_PL = _read(_BACKEND, "routers", "plans.py")
_TL = _read(_BACKEND, "tools.py")
_CT = _read(_BACKEND, "cron_tasks.py")


# ---------------------------------------------------------------------------
# P2-3 · flip del swap C→P
# ---------------------------------------------------------------------------

def test_p2_3_carb_to_protein_swap_default_on():
    assert re.search(
        r'CARB_TO_PROTEIN_SWAP_ENABLED\s*=\s*_env_bool\("MEALFIT_CARB_TO_PROTEIN_SWAP",\s*True\)', _GO
    ), "el swap C→P validado por A/B debe estar ON en código (vivía solo en el .env del VPS)"


# ---------------------------------------------------------------------------
# P2-1 / P2-2 · micros post-motor
# ---------------------------------------------------------------------------

def test_p2_1_knobs_and_wiring():
    assert '_env_bool("MEALFIT_MICRO_POSTENGINE_RECOMPUTE", True)' in _GO
    # el recompute post-motor corre en assemble DESPUÉS del qty-sync final (ventana acotada).
    _sync = _GO.index("mención(es) de cantidad en pasos re-sincronizadas post-quantize")
    _win = _GO[_sync:_sync + 4500]
    assert "recompute_micronutrient_report_for_plan(result, form_data)" in _win, \
        "falta el recompute del panel post-motor en assemble (P2-1)"
    # y también en la convergencia de presupuesto (las sustituciones cambian micros).
    _conv = _GO.index('apply_update_macro_engine(result, surface="budget_convergence"')
    assert "recompute_micronutrient_report_for_plan(result, form_data, db=_bc_db)" in _GO[_conv:_conv + 900]


def test_p2_2_postengine_recheck_block():
    assert '_env_bool("MEALFIT_MICRO_POSTENGINE_RECHECK", True)' in _GO
    _blk_start = _GO.index("(P2-2) Micro-recheck post-motor")
    _blk = _GO[_blk_start:_blk_start + 2200]
    assert "_close_micro_gaps_for_plan(result, form_data, _pe_db)" in _blk
    assert "_trim_day_carbs_to_target" in _blk, "el re-trim preserva el cierre (patrón BAND-RECHECK)"
    assert "_apply_portion_quantization" in _blk
    assert "_sync_recipe_step_quantities" in _blk


# ---------------------------------------------------------------------------
# P2-5 · gate suave de transformación
# ---------------------------------------------------------------------------

def test_p2_5_transform_gate_knobs_and_shape():
    assert '_env_bool("MEALFIT_TRANSFORM_SOFT_GATE", True)' in _GO
    assert '_env_int("MEALFIT_TRANSFORM_GATE_MIN_COUNT", 1' in _GO
    _gate = _GO.index("(P2-5) Gate SUAVE de TRANSFORMACIÓN culinaria")
    _blk = _GO[_gate:_gate + 2600]
    assert '"_transform_gate_advisory_final"' in _blk, "advisory en intento final (nunca cero-plan)"
    assert "approved = False" in _blk, "en intentos 1..N-1 debe rechazar para retry"
    # corre DESPUÉS del gate raw-staple (mismo orden del review).
    assert _GO.index("Gate SUAVE de creatividad") < _gate


# ---------------------------------------------------------------------------
# P2-7 · clamp timetemp en todos los pasos (funcional)
# ---------------------------------------------------------------------------

@pytest.fixture()
def _go_mod():
    import graph_orchestrator as go
    return go


def test_p2_7_clamps_non_toque_steps(_go_mod):
    meal = {
        "name": "Pollo horneado",
        "recipe": [
            "Mise en place: corta el pollo.",
            "El toque de fuego: hornea 18-20 min a 180 °C.",
            "Montaje: gratina el queso al horno por 200 min a 300 °C antes de servir.",
        ],
    }
    changed = _go_mod._clamp_recipe_time_temp_outliers(meal)
    assert changed is True
    _montaje = meal["recipe"][2]
    assert "200 min" not in _montaje, "el tiempo absurdo fuera del Toque de Fuego debe clamparse"
    assert "300 °C" not in _montaje and "220 °C" in _montaje
    assert meal.get("_recipe_timetemp_clamped") is True


def test_p2_7_passive_steps_time_exempt_temp_clamped(_go_mod):
    meal = {
        "name": "Pollo al horno",
        "recipe": [
            "Mise en place: marina el pollo por 4 horas en la nevera.",
            "El toque de fuego: hornea 18-20 min a 180 °C.",
        ],
    }
    changed = _go_mod._clamp_recipe_time_temp_outliers(meal)
    assert changed is False, "marinar 4 horas es legítimo — el paso pasivo no se toca"
    assert "4 horas" in meal["recipe"][0]

    meal2 = {"name": "X", "recipe": ["Deja reposar la masa 1 hora junto al horno a 400 grados."]}
    _go_mod._clamp_recipe_time_temp_outliers(meal2)
    assert "1 hora" in meal2["recipe"][0], "el TIEMPO pasivo queda exento"
    assert "400" not in meal2["recipe"][0], "la TEMPERATURA absurda se clampa incluso en paso pasivo"


def test_p2_7_idempotente_y_knob_off(_go_mod, monkeypatch):
    meal = {"name": "Y", "recipe": ["El toque de fuego: hornea 18-20 min a 180 °C."]}
    assert _go_mod._clamp_recipe_time_temp_outliers(meal) is False
    monkeypatch.setattr(_go_mod, "RECIPE_TIMETEMP_PLAUSIBILITY_ENABLED", False)
    meal_bad = {"name": "Z", "recipe": ["Montaje: hornea 500 min."]}
    assert _go_mod._clamp_recipe_time_temp_outliers(meal_bad) is False


# ---------------------------------------------------------------------------
# P2-12 · alert en approved-con-residuo
# ---------------------------------------------------------------------------

def test_p2_12_alert_wired_and_documented():
    assert '_env_bool("MEALFIT_APPROVED_RESIDUAL_ALERT", True)' in _GO
    _blk_start = _GO.index("(P2-12) El usuario ya veía el banner en este path")
    _blk = _GO[_blk_start:_blk_start + 900]
    assert '_emit_plan_quality_degraded_alert(state, "approved_with_residual", severity="minor")' in _blk
    _doc = _read(_BACKEND, "docs", "system_alerts_resolution_table.md")
    assert "approved_with_residual" in _doc, "la fila del doc canónico debe reflejar el 6º emit"


# ---------------------------------------------------------------------------
# P2-4 · expand con paridad de motor
# ---------------------------------------------------------------------------

def test_p2_4_expand_engine_and_band_parity_order():
    _eng = _PL.index('_ume_exp(plan_data_fresh, surface="recipe_expand")')
    _mic = _PL.index("from graph_orchestrator import recompute_micronutrient_report_for_plan as _rmr_exp")
    _bp = _PL.index('_ubp_exp(plan_data_fresh, surface="recipe_expand")')
    assert _eng < _mic < _bp, "orden del contrato de updates: motor → micros → band-parity"


# ---------------------------------------------------------------------------
# P2-10 · variedad entre días (mismo slot) en chat-modify
# ---------------------------------------------------------------------------

def test_p2_10_chat_modify_cross_day_slot_names():
    _blk_start = _TL.index("(P2-10) Paridad exacta con el swap")
    _blk = _TL[_blk_start:_blk_start + 1600]
    assert "VARIEDAD ENTRE DÍAS" in _blk
    assert "_slot_names_cd" in _blk
    # el filtro es por MISMO slot (paridad con _cross_day_meal_names_for_swap del swap).
    assert 'str(_om.get("meal", "")).lower().strip() == _mt_cd' in _blk


# ---------------------------------------------------------------------------
# P2-11 · contrato de sincronía regen-day ↔ helper SSOT
# ---------------------------------------------------------------------------

def test_p2_11_sync_contract_banda_and_step():
    # Ambas implementaciones comparten banda [0.90, 1.12] y step de 5g.
    def _fn_block(src, anchor, span=9000):
        i = src.index(anchor)
        return src[i:i + span]

    helper = _fn_block(_GO, "def apply_update_macro_engine(")
    inline = _fn_block(_PL, "refine_day_portions_integer as _rdi_rd", span=3000)
    for token in ("0.90", "1.12"):
        assert token in helper, f"banda {token} ausente del helper SSOT"
        assert token in _fn_block(_PL, "_oob_rf = any", span=400), f"banda {token} ausente del inline regen-day"
    assert "step_g=5.0" in helper and "step_g=5.0" in inline, "step de refine divergió (5g)"
    # el inline importa los knobs SSOT (no copia valores).
    for knob in ("PORTION_SHRINK_FLOOR_G", "PORTION_CAP_PROTEIN_G",
                 "_SHRINK_FLOOR_EXEMPT_TOKENS", "GLOBAL_DAY_REFINE_MAX_ITERS"):
        assert knob in _PL, f"regen-day dejó de importar el knob SSOT {knob}"
    # cross-links vivos en ambos sitios.
    assert "test_p2_audit_v7_batch.py" in helper
    assert "CONTRATO DE SINCRONÍA" in _PL


# ---------------------------------------------------------------------------
# P2-6 · cross-semana determinista (funcional + wiring)
# ---------------------------------------------------------------------------

def _mk_days_xw():
    return [
        {"day": 1, "meals": [
            {"meal": "Cena", "name": "Pollo guisado con yuca"},
            {"meal": "Almuerzo", "name": "Locrio de pollo"},
        ]},
        {"day": 8, "meals": [
            {"meal": "Cena", "name": "Pollo Guisado con Yuca"},   # repite slot+nombre (case/acentos)
            {"meal": "Almuerzo", "name": "Moro de habichuelas"},  # nuevo
            {"meal": "Desayuno", "name": "Locrio de pollo"},      # mismo nombre pero OTRO slot → no repite
        ]},
    ]


def test_p2_6_cross_week_report_functional(_go_mod):
    days = _mk_days_xw()
    rep = _go_mod.compute_cross_week_repeat_report(days, [8])
    assert rep["checked_meals"] == 3
    assert rep["repeat_count"] == 1
    assert rep["repeats"][0]["slot"] == "Cena"
    assert rep["repeats"][0]["repeats_days"] == [1]
    _cena_w2 = days[1]["meals"][0]
    assert _cena_w2.get("_cross_week_repeat") is True
    _des_w2 = days[1]["meals"][2]
    assert "_cross_week_repeat" not in _des_w2, "mismo nombre en OTRO slot no cuenta (conservador)"


def test_p2_6_no_new_days_or_knob_off(_go_mod, monkeypatch):
    assert _go_mod.compute_cross_week_repeat_report(_mk_days_xw(), []) == {}
    monkeypatch.setattr(_go_mod, "CROSS_WEEK_REPEAT_CHECK_ENABLED", False)
    assert _go_mod.compute_cross_week_repeat_report(_mk_days_xw(), [8]) == {}


def test_p2_6_wired_in_chunk_merge():
    _blk_start = _CT.index("compute_cross_week_repeat_report as _xwr_ck")
    _blk = _CT[_blk_start:_blk_start + 1800]
    assert '_cross_week_repeat_report"] = _xw_rep' in _blk or 'plan_data["_cross_week_repeat_report"]' in _blk
    # corre en el merge T1, tras el dish-quality report y antes del recompute de micros.
    assert _CT.index("(P2-C) dish_quality_report también en chunk T1") < _blk_start
    assert _blk_start < _CT.index("Recalcula el panel de micros")


# ---------------------------------------------------------------------------
# P2-8 / P2-9 / chip · frontend
# ---------------------------------------------------------------------------

_HAS_FRONTEND = os.path.isdir(_FRONTEND)


@pytest.mark.skipif(not _HAS_FRONTEND, reason="workspace sin frontend/ (repo backend aislado)")
def test_p2_8_shopping_list_panel_component():
    panel = _read(_FRONTEND, "src", "components", "dashboard", "ShoppingListPanel.jsx")
    assert "is_perishable" in panel, "prioridad 1: flag SSOT del backend"
    assert "shelf_life_days" in panel, "prioridad 2: umbral shelf_life (mismo que backend)"
    assert "display_category" in panel, "categoría = display_category SSOT"
    dash = _read(_FRONTEND, "src", "pages", "Dashboard.jsx")
    assert "ShoppingListPanel" in dash and "<ShoppingListPanel" in dash, "el panel debe montarse en Dashboard"


@pytest.mark.skipif(not _HAS_FRONTEND, reason="workspace sin frontend/ (repo backend aislado)")
def test_p2_9_pantry_limited_cta():
    dash = _read(_FRONTEND, "src", "pages", "Dashboard.jsx")
    assert "_quality_degraded_pantry_limited" in dash, \
        "la señal P1-PANTRY-DEGRADED-SIGNAL debe tener lector en la UI"
    _cta = dash.index("_quality_degraded_pantry_limited")
    assert "navigate('/dashboard/pantry')" in dash[_cta:_cta + 3000], "CTA debe llevar a la Nevera"


@pytest.mark.skipif(not _HAS_FRONTEND, reason="workspace sin frontend/ (repo backend aislado)")
def test_p2_6_chip_cross_week_in_meal_advisories():
    adv = _read(_FRONTEND, "src", "utils", "mealAdvisories.js")
    assert "_cross_week_repeat" in adv
    assert "cross_week_repeat" in adv


# ---------------------------------------------------------------------------
# Marker bump
# ---------------------------------------------------------------------------

def test_marker_bumped():
    app = _read(_BACKEND, "app.py")
    assert '_LAST_KNOWN_PFIX = "P2-AUDIT-V7-BATCH · 2026-07-04"' in app
