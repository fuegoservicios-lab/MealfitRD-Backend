"""[P1-NEXT-LEVEL-BATCH · 2026-07-02] Las 4 features "siguiente nivel" del motor.

  #4 GYM     — plan_gym.score_plan: scoring multi-eje (8 ejes) de un plan entregado
               + CLI multi-perfil (scripts/plan_gym.py, reusa los 20 perfiles held-out).
  #1 TASTE   — taste_model: preferencias APRENDIDAS del uso (swap-away/chat-replace →
               user_taste_events → contexto suave al planner/day-gen vía taste_profile).
  #3 LIBRARY — dish_library + data/dish_templates.json (~85 plantillas DD curadas):
               el day-gen ELIGE Y ADAPTA (creatividad por recombinación), prioriza
               transformadas (panqueques/bollitos/pastelón), slots SSOT-compliant.
  #2 SOLVER  — portion_solver.refine_day_portions_integer: refinador GLOBAL entero del
               día (pasos 5g post-quantize, kcal+P+C+F conjuntos, sin re-quantize).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_PL = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_TO = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_DG = (_BACKEND / "prompts" / "day_generator.py").read_text(encoding="utf-8")


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-NEXT-LEVEL-BATCH" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-02"


# ════════════════════════════════════════════════════════════════════════════
# #4 GYM — scoring multi-eje
# ════════════════════════════════════════════════════════════════════════════
def _good_plan():
    return {
        "calories": "2000 kcal", "macros": {"protein": "150g", "carbs": "220g", "fats": "60g"},
        "days": [{"day": 1, "meals": [
            {"meal": "Desayuno", "name": "Mangú con Huevos Revueltos", "cals": 500, "protein": 38, "carbs": 55, "fats": 15,
             "ingredients": ["2 huevos", "200g de plátano"], "recipe": ["Mise en place: pela.", "El Toque de Fuego: hierve 12-15 min.", "Montaje: sirve."]},
            {"meal": "Almuerzo", "name": "Locrio de Pollo", "cals": 700, "protein": 55, "carbs": 80, "fats": 20,
             "ingredients": ["150g de pollo", "80g de arroz"], "recipe": ["Mise en place: corta.", "El Toque de Fuego: guisa 20-25 min a fuego medio.", "Montaje: sirve."]},
            {"meal": "Cena", "name": "Pescado con Puré de Yuca", "cals": 800, "protein": 57, "carbs": 85, "fats": 25,
             "ingredients": ["180g de pescado", "200g de yuca"], "recipe": ["Mise en place: sazona.", "El Toque de Fuego: plancha 3-4 min por lado.", "Montaje: sirve."]},
        ]}],
        "micronutrient_report": {"gaps": [], "coverage": 0.9,
                                 "per_day_floors": {"flagged": False}, "per_day_ceilings": {"flagged": False}},
        "budget_reconciliation": {"status": "dentro", "ratio": 0.9},
    }


def test_gym_scores_good_plan_high():
    from plan_gym import score_plan
    s = score_plan(_good_plan(), {})
    assert s["global"] >= 85, f"plan bueno debe puntuar alto (got {s['global']})"
    assert s["axes"]["banda"] == 100.0 and s["axes"]["slots"] == 100.0
    assert s["axes"]["entrega"] == 100.0


def test_gym_penalizes_violations_and_caps_fallback():
    import copy
    from plan_gym import score_plan
    bad = copy.deepcopy(_good_plan())
    bad["days"][0]["meals"][2]["name"] = "Arroz Blanco con Salami"   # arroz de noche (soft)
    bad["_is_fallback"] = True
    s = score_plan(bad, {})
    assert s["axes"]["slots"] < 100.0, "la violación de slot debe penalizar"
    assert s["axes"]["entrega"] == 0.0
    assert s["global"] <= 40.0, "un plan fallback queda capeado (la entrega manda)"


def test_gym_missing_axes_excluded_not_zeroed():
    from plan_gym import score_plan
    p = _good_plan()
    p.pop("micronutrient_report")
    p.pop("budget_reconciliation")
    s = score_plan(p, {})
    assert s["axes"]["micros"] is None and s["axes"]["presupuesto"] is None
    assert s["global"] > 80, "ejes sin datos se excluyen del promedio, no lo hunden"


def test_gym_aggregate():
    from plan_gym import score_plan, aggregate_scores
    s = score_plan(_good_plan(), {})
    agg = aggregate_scores([{"id": 1, "score": s}, {"id": 2, "score": s}])
    assert agg["n"] == 2 and agg["global_mean"] == s["global"]
    assert "banda" in agg and agg["banda"]["mean"] == 100.0


# ════════════════════════════════════════════════════════════════════════════
# #1 TASTE — modelo de gustos aprendido
# ════════════════════════════════════════════════════════════════════════════
def test_taste_protein_token_and_guest_guard():
    import taste_model as tm
    assert tm.protein_token_of("Pechuga a la Plancha con Yuca") == "pollo"
    assert tm.protein_token_of("Fresas con Yogur") is None  # anti 'res'-en-'fresas'
    assert tm._is_real_user("guest") is False
    assert tm._is_real_user(None) is False
    assert tm._is_real_user("3f2b8c1a-9d4e-4f6a-b7c8-1a2b3c4d5e6f") is True


def test_taste_swap_signal_only_on_protein_change(monkeypatch):
    import taste_model as tm
    recorded = []
    monkeypatch.setattr(tm, "_record", lambda *a: recorded.append(a) or True)
    uid = "3f2b8c1a-9d4e-4f6a-b7c8-1a2b3c4d5e6f"
    # misma proteína → cero señal (el swap no era sobre ella)
    assert tm.record_swap_away(uid, "Pollo Guisado", "Pechuga al Horno") is False
    # proteína distinta → señal débil sobre la vieja
    assert tm.record_swap_away(uid, "Pollo Guisado", "Res a la Plancha") is True
    assert recorded[-1][1] == "pollo" and recorded[-1][3] == 1.0


def test_taste_chat_strong_negative(monkeypatch):
    import taste_model as tm
    recorded = []
    monkeypatch.setattr(tm, "_record", lambda *a: recorded.append(a) or True)
    uid = "3f2b8c1a-9d4e-4f6a-b7c8-1a2b3c4d5e6f"
    assert tm.record_chat_replace(uid, "Pechuga Guisada", "Res Mechada", "no me gusta el pollo, cámbialo")
    assert recorded[-1][2] == "chat_negative" and recorded[-1][3] == 2.0
    # negación sin mencionar el token → NO fuerte (débil por cambio de proteína)
    recorded.clear()
    assert tm.record_chat_replace(uid, "Pechuga Guisada", "Res Mechada", "no me gusta este plato")
    assert recorded[-1][2] == "chat_replace" and recorded[-1][3] == 1.0


def test_taste_context_empty_without_signals(monkeypatch):
    import taste_model as tm
    monkeypatch.setattr(tm, "negative_tokens_for_user", lambda uid: [])
    assert tm.build_taste_context("3f2b8c1a-9d4e-4f6a-b7c8-1a2b3c4d5e6f") == "", \
        "sin señal → '' (byte-equivalente, preserva prompt-cache)"
    monkeypatch.setattr(tm, "negative_tokens_for_user", lambda uid: ["pavo"])
    ctx = tm.build_taste_context("3f2b8c1a-9d4e-4f6a-b7c8-1a2b3c4d5e6f")
    assert "pavo" in ctx and "NO alergia" in ctx


def test_taste_wired_capture_and_injection():
    # captura en swap-persist + chat-modify; inyección en el context builder
    assert "record_swap_away" in _PL and "_taste_old_name" in _PL
    assert "record_chat_replace" in _TO
    assert "build_taste_context as _btc_learned" in _GO
    # migración SSOT en ambos dirs
    assert (_BACKEND / "migrations" / "p1_user_taste_events_2026_07_02.sql").exists()
    assert (_BACKEND.parent / "migrations" / "p1_user_taste_events_2026_07_02.sql").exists()


# ════════════════════════════════════════════════════════════════════════════
# #3 LIBRARY — biblioteca de platos
# ════════════════════════════════════════════════════════════════════════════
def test_library_loads_and_is_slot_compliant():
    from dish_library import load_dish_templates
    ts = load_dish_templates()
    assert len(ts) >= 70, f"biblioteca curada esperaba ≥70 plantillas (got {len(ts)})"
    # SSOT de slots: cero arroz/locrio/moro/pasta/sopa en desayuno o cena
    _rice_re = re.compile(r"\b(arroz|locrio|moro|asopao|espagueti|sancocho|sopa)\b", re.I)
    bad = [t["name"] for t in ts
           if any(s in ("desayuno", "cena") for s in t["slots"])
           and _rice_re.search(t["name"]) and "harina de arroz" not in t["name"].lower()]
    assert not bad, f"plantillas violan el SSOT de slots: {bad}"
    # el valor central: una fracción sustancial son TRANSFORMADAS
    transformed = sum(1 for t in ts if t.get("transform"))
    assert transformed >= 30, f"esperaba ≥30 transformadas (got {transformed})"


def test_library_context_deterministic_and_pool_filtered():
    from prompts.day_generator import build_day_assignment_context
    sk = {"protein_pool": ["Pescado fresco"], "carb_pool": ["Yuca"], "fruit_pool": ["Guineo"],
          "meal_types": ["Cena"], "assigned_technique": "plancha", "brief_concept": "Mar"}
    c1 = build_day_assignment_context(sk, 3)
    assert c1 == build_day_assignment_context(sk, 3), "el bloque debe ser determinista (prompt-cache)"
    assert "INSPIRACIÓN DOMINICANA" in c1
    cena_line = next((l for l in c1.splitlines() if "Cena:" in l), "")
    assert "Locrio" not in cena_line and "Moro" not in cena_line
    # pool de pescado → sin platos de res/cerdo como principal en la muestra de cena
    assert not re.search(r"\b(res|cerdo|chuleta)\b", cena_line, re.I), cena_line


def test_library_knob_off_removes_block(monkeypatch):
    import dish_library as dl
    monkeypatch.setattr(dl, "DISH_LIBRARY_ENABLED", False)
    assert dl.build_dish_library_context({"protein_pool": ["Pollo"], "meal_types": ["Cena"]}, 1) == ""


def test_library_wired_in_day_generator():
    assert "build_dish_library_context" in _DG


# ════════════════════════════════════════════════════════════════════════════
# #2 SOLVER — refinador global entero
# ════════════════════════════════════════════════════════════════════════════
_DENS = {"pollo": {"kcal": 1.65, "protein": 0.31, "carbs": 0.0, "fats": 0.036},
         "arroz": {"kcal": 1.30, "protein": 0.027, "carbs": 0.28, "fats": 0.003},
         "aguacate": {"kcal": 1.60, "protein": 0.02, "carbs": 0.085, "fats": 0.147}}


class _RefDB:
    def macros_from_ingredient_string(self, s):
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*g de (\w+)", str(s).lower())
        if not m:
            return None
        g, food = float(m.group(1)), m.group(2)
        d = _DENS.get(food)
        return {k: v * g for k, v in d.items()} if d else None


def _ref_meals():
    return [
        {"name": "Almuerzo", "ingredients": ["120g de pollo", "80g de arroz", "50g de aguacate"],
         "ingredients_raw": ["120g de pollo", "80g de arroz", "50g de aguacate"]},
        {"name": "Cena", "ingredients": ["100g de pollo", "150g de arroz"],
         "ingredients_raw": ["100g de pollo", "150g de arroz"]},
    ]


def test_refiner_converges_all4_with_human_steps():
    from portion_solver import refine_day_portions_integer
    meals = _ref_meals()
    targets = {"protein": 90.0, "carbs": 70.0, "fats": 22.0, "kcal": 4 * 160 + 9 * 22}
    db = _RefDB()
    n = refine_day_portions_integer(meals, targets, db, step_g=5.0, floor_g=15.0, cap_g=300.0)
    assert n > 0
    delivered = {"kcal": 0.0, "protein": 0.0, "carbs": 0.0, "fats": 0.0}
    grams = []
    for m in meals:
        for s in m["ingredients"]:
            mc = db.macros_from_ingredient_string(s)
            for k in delivered:
                delivered[k] += mc[k]
            grams.append(float(re.match(r"(\d+(?:\.\d+)?)", s).group(1)))
    assert all(g % 5 == 0 for g in grams), f"las porciones deben seguir humanas (múltiplos de 5g): {grams}"
    for k in ("protein", "carbs", "fats"):
        assert 0.90 <= delivered[k] / targets[k] <= 1.12, f"{k} fuera de banda: {delivered[k] / targets[k]:.3f}"
    assert 0.90 <= delivered["kcal"] / targets["kcal"] <= 1.12


def test_refiner_respects_bounds_and_lockstep():
    from portion_solver import refine_day_portions_integer
    meals = _ref_meals()
    targets = {"protein": 500.0, "carbs": 70.0, "fats": 22.0, "kcal": 3000.0}  # inalcanzable
    refine_day_portions_integer(meals, targets, _RefDB(), step_g=5.0, floor_g=15.0, cap_g=300.0)
    for m in meals:
        assert m["ingredients"] == m["ingredients_raw"], "lockstep raw debe mantenerse"
        for s in m["ingredients"]:
            g = float(re.match(r"(\d+(?:\.\d+)?)", s).group(1))
            assert g <= 300.0, "cap absoluto respetado"
            assert g >= 15.0, "piso respetado"


def test_refiner_noop_cases():
    from portion_solver import refine_day_portions_integer
    assert refine_day_portions_integer([], {"protein": 100}, _RefDB()) == 0
    assert refine_day_portions_integer(_ref_meals(), {}, _RefDB()) == 0
    # ya en banda → puede refinar poco o nada, pero jamás explotar
    meals = _ref_meals()
    n = refine_day_portions_integer(
        meals, {"protein": 75.4, "carbs": 68.7, "fats": 16.0, "kcal": 742.0}, _RefDB())
    assert isinstance(n, int)


def test_refiner_wired_in_assemble():
    assert "GLOBAL_DAY_REFINE_ENABLED" in _GO
    assert "refine_day_portions_integer as _rdi" in _GO
    # corre ANTES del recheck (fallback continuo) y del qty-sync final
    refine_idx = _GO.index("refine_day_portions_integer as _rdi")
    recheck_idx = _GO.index("POSTQUANTIZE_RECHECK_ENABLED and (_pg or _cg or _fg)")
    assert refine_idx < recheck_idx, "el refinador global debe correr antes del recheck continuo"
