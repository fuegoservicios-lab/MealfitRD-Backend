"""[P2-UPDATE-INTELLIGENCE-2 · 2026-06-24] Regresión de los 5 P2 de la RE-auditoría de inteligencia.

  P2-1 FOOD-SAFETY: `food_safety_backstop_for_meal` (reusa `_apply_food_safety_fixes` de S1) re-aplica la
       nota de seguridad (huevo/pescado-marisco crudos) en swap/regenerate-day/chat-modify. Default ON
       (FOOD_SAFETY_GUARD), macro-preservante, fail-open.
  P2-2 PROTEIN-CLOSER: closer de proteína per-meal en swap (reusa `_close_protein_gap_for_meal`), con
       re-validación de pantry + revert. Default OFF (A/B con P1-2). regenerate-day (P2-3) lo hereda.
  P2-4 DISLIKES: `_enrich_clinical_from_profile` hidrata dislikes server-side (UNION body+perfil),
       espejo de allergies. Default ON (MEALFIT_UPDATE_HYDRATE_DISLIKES).
  P2-5 PANTRY-OVERRIDE: regenerate-day pasa `pantry_override` → swap_meal valida contra el ledger
       reservado (reserva inter-plato D7), no contra la nevera-virtual completa. Default ON.

Parser-based (corre local + CI) + funcional guardado por import de graph_orchestrator / routers.plans (CI).
"""
import ast
import os
import re

import pytest

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(rel):
    with open(os.path.join(BACKEND, rel), encoding="utf-8") as f:
        return f.read()


def _func_src(source, name):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(source, node)
    raise AssertionError(f"función {name!r} no encontrada")


AGENT = _read("agent.py")
TOOLS = _read("tools.py")
PLANS = _read("routers/plans.py")
ORCH = _read("graph_orchestrator.py")
APP = _read("app.py")


# ── P2-1: food-safety en updates ──────────────────────────────────────────────
def test_p2_1_food_safety_helper_exists_and_reuses_s1():
    assert "def food_safety_backstop_for_meal" in ORCH
    src = _func_src(ORCH, "food_safety_backstop_for_meal")
    assert "FOOD_SAFETY_GUARD" in src, "debe gatearse por FOOD_SAFETY_GUARD (el mismo de S1)"
    assert "_apply_food_safety_fixes" in src, "debe reusar el fix determinista de S1 (SSOT)"


def test_p2_1_food_safety_wired_in_swap_and_modify():
    assert "food_safety_backstop_for_meal" in _func_src(AGENT, "swap_meal"), "swap debe correr food-safety"
    assert "food_safety_backstop_for_meal" in _func_src(TOOLS, "execute_modify_single_meal"), "modify debe correr food-safety"


# ── P2-2 / P2-3: closer de proteína per-meal (default OFF, A/B) ────────────────
def test_p2_2_swap_protein_closer_wired_off_with_pantry_revert():
    src = _func_src(AGENT, "swap_meal")
    assert "MEALFIT_SWAP_PER_MEAL_MACRO_CLOSER" in src
    assert re.search(r'MEALFIT_SWAP_PER_MEAL_MACRO_CLOSER["\']\s*,\s*["\']false', src), "P2-2 default OFF (A/B con P1-2)"
    assert "_close_protein_gap_for_meal" in src, "debe reusar el closer determinista de S1"
    # re-validación pantry + revert (never-worse-than-current)
    assert "validate_ingredients_against_pantry" in src and "_snap_cl" in src
    # renal exento (el trim renal manda)
    assert "not _renal_capped" in src


# ── P2-4: dislikes hidratados server-side ─────────────────────────────────────
def test_p2_4_enrich_hydrates_dislikes_union():
    src = _func_src(PLANS, "_enrich_clinical_from_profile")
    assert "MEALFIT_UPDATE_HYDRATE_DISLIKES" in src
    assert 'data["dislikes"]' in src, "el enrich debe asignar dislikes (UNION body+perfil)"
    assert 'hp.get("dislikes")' in src, "debe leer dislikes del health_profile"


# ── P2-5: pantry override (ledger reservado) ──────────────────────────────────
def test_p2_5_regenerate_day_sets_pantry_override():
    src = _func_src(PLANS, "api_regenerate_day")
    assert '"pantry_override": True' in src, "regenerate-day debe señalar pantry_override al swap"


def test_p2_5_swap_honors_pantry_override():
    src = _func_src(AGENT, "swap_meal")
    assert "MEALFIT_REGEN_DAY_PANTRY_OVERRIDE" in src
    assert re.search(r'MEALFIT_REGEN_DAY_PANTRY_OVERRIDE["\']\s*,\s*["\']true', src), "P2-5 default ON"
    assert "pantry_override" in src and "_override_lines" in src


# ── marker + knobs ────────────────────────────────────────────────────────────
def test_p2_marker_bumped():
    # [de-pin · 2026-06-26] `_LAST_KNOWN_PFIX` es single-valued → pinear "P2-UPDATE-INTELLIGENCE-2"
    # quedó stale apenas un P-fix posterior bumpeó el marker. Contrato durable del bump:
    # test_p3_1_last_known_pfix_freshness (formato + floor) + test_p2_hist_audit_14_marker_test_link.
    assert re.search(r'_LAST_KNOWN_PFIX\s*=\s*"P\d+-[A-Z0-9-]+ · \d{4}-\d{2}-\d{2}"', APP), \
        "_LAST_KNOWN_PFIX debe existir con formato `Pn-... · YYYY-MM-DD`"


def test_p2_knobs_present():
    for knob in ("MEALFIT_SWAP_PER_MEAL_MACRO_CLOSER", "MEALFIT_UPDATE_HYDRATE_DISLIKES",
                 "MEALFIT_REGEN_DAY_PANTRY_OVERRIDE"):
        assert knob in (AGENT + PLANS + ORCH), f"knob {knob} ausente"


# ── Funcional: food-safety (guardado por import graph_orchestrator) ───────────
try:
    import graph_orchestrator as _GO
    _GO_ERR = None
except Exception as _e:  # pragma: no cover
    _GO = None
    _GO_ERR = _e

requires_go = pytest.mark.skipif(_GO is None, reason=f"graph_orchestrator no importable: {_GO_ERR}")


@requires_go
def test_food_safety_backstop_flags_ceviche(monkeypatch):
    monkeypatch.setattr(_GO, "FOOD_SAFETY_GUARD", True)
    monkeypatch.setattr(_GO, "RAW_SEAFOOD_SAFETY_ENABLED", True)
    meal = {"name": "Ceviche de Pescado", "ingredients": ["Pescado fresco", "Limón"], "recipe": ["Marinar en limón"]}
    assert _GO.food_safety_backstop_for_meal(meal) >= 1
    assert meal.get("_food_safety_seafood") is True
    # idempotente: 2ª pasada no re-mitiga
    assert _GO.food_safety_backstop_for_meal(meal) == 0


@requires_go
def test_food_safety_backstop_noop_when_gated_off(monkeypatch):
    monkeypatch.setattr(_GO, "FOOD_SAFETY_GUARD", False)
    assert _GO.food_safety_backstop_for_meal({"name": "Ceviche de Pescado"}) == 0
    # no-op para input no-dict
    monkeypatch.setattr(_GO, "FOOD_SAFETY_GUARD", True)
    assert _GO.food_safety_backstop_for_meal(None) == 0


# ── Funcional: dislikes (guardado por import routers.plans) ────────────────────
try:
    from routers.plans import _enrich_clinical_from_profile as _ENRICH
    _ENRICH_ERR = None
except Exception as _e:  # pragma: no cover
    _ENRICH = None
    _ENRICH_ERR = _e

requires_router = pytest.mark.skipif(_ENRICH is None, reason=f"routers.plans no importable: {_ENRICH_ERR}")


@requires_router
def test_enrich_hydrates_dislikes_from_profile(monkeypatch):
    import db
    monkeypatch.setattr(db, "get_user_profile", lambda uid: {"health_profile": {
        "allergies": [], "dietType": "balanced", "dislikes": ["berenjena", "hígado"],
    }})
    data = {"user_id": "u1", "dislikes": []}  # body vacío (ventana stale)
    _ENRICH(data, "u1")
    assert set(data.get("dislikes") or []) == {"berenjena", "hígado"}, "debe rellenar dislikes del perfil"
