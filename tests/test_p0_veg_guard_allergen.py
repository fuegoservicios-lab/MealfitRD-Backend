"""[P0-VEG-GUARD-ALLERGEN · 2026-07-01] (audit P0-2 · seguridad de alérgenos en updates)

El veg-guard (`_add_missing_recipe_step_vegetables`) INYECTA ingredientes ("100g Apio") cuando un
vegetal aparece en los PASOS pero no en ingredients[]. En form-gen corre ANTES del allergen scan
(el scan posterior lo cubre), pero en las superficies de UPDATE (swap/chat-modify/recalculate/
recipe-expand) el finalizer corre DESPUÉS del backstop clínico → sin filtro, el guard podía
MATERIALIZAR un alérgeno IgE post-scan (alergia a apio/tomate + paso que los menciona).

Fix: parámetro `allergies` en el veg-guard que EXCLUYE candidatos que disparan
`_scan_allergen_violations` (SSOT sinónimos DD, fail-secure per-candidato), cableado desde:
  - swap (agent.py, `allergies` enriquecidas server-side) → regenerate-day lo hereda
  - chat-modify (tools.py, `_clin_allergies` del perfil)
  - /recalculate-shopping-list (plans.py, hidratación propia del perfil)
  - /recipe/expand (plans.py, `_expand_allergies` vía _enrich_clinical_from_profile)
  - finalize_plan_data_coherence (pass-through opcional para callers futuros)

Tests: (1) funcional del filtro con catálogo mockeado (sin Neon); (2) fail-secure;
(3) parser-based del wiring en las superficies.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import graph_orchestrator as g

_BACKEND = Path(__file__).resolve().parent.parent
_AGENT = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_TOOLS = (_BACKEND / "tools.py").read_text(encoding="utf-8")
_PLANS = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_GRAPH = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")


def _fake_rows():
    return [
        {"name": "Apio", "category": "Vegetales", "kcal_per_100g": 16},
        {"name": "Brócoli", "category": "Vegetales", "kcal_per_100g": 34},
    ]


def _patch_catalog(monkeypatch):
    import shopping_calculator as sc
    monkeypatch.setattr(sc, "get_master_ingredients", _fake_rows)
    # normalize_name real puede tocar fuzzy/embeddings; para el guard basta un normalizador simple.
    monkeypatch.setattr(sc, "normalize_name", lambda s: str(s).strip().lower())


# ---------------------------------------------------------------------------
# 1. Funcional: el filtro excluye el alérgeno y conserva el resto
# ---------------------------------------------------------------------------
def test_veg_guard_excludes_declared_allergen(monkeypatch):
    _patch_catalog(monkeypatch)
    days = [{"meals": [{"name": "Guiso", "ingredients": [],
                        "recipe": ["Agrega el apio y el brocoli picados y saltea 5 min."]}]}]
    n = g._add_missing_recipe_step_vegetables(days, allergies=["apio"])
    ings = " ".join(str(x) for x in days[0]["meals"][0]["ingredients"]).lower()
    assert "apio" not in ings, "el veg-guard NO debe inyectar un alérgeno declarado"
    assert "brocoli" in ings or "brócoli" in ings, "los candidatos no-alérgenos siguen añadiéndose"
    assert n == 1


def test_veg_guard_unchanged_without_allergies(monkeypatch):
    """Sin alergias (default None / []), el comportamiento pre-fix se preserva."""
    _patch_catalog(monkeypatch)
    days = [{"meals": [{"name": "Guiso", "ingredients": [],
                        "recipe": ["Agrega el apio y el brocoli picados y saltea 5 min."]}]}]
    assert g._add_missing_recipe_step_vegetables(days) == 2
    days2 = [{"meals": [{"name": "Guiso", "ingredients": [],
                         "recipe": ["Agrega el apio y el brocoli picados y saltea 5 min."]}]}]
    assert g._add_missing_recipe_step_vegetables(days2, allergies=[]) == 2


def test_veg_guard_fail_secure_on_scanner_error(monkeypatch):
    """Si el scanner de alérgenos crashea, el candidato NO se añade (fail-secure per-candidato)."""
    _patch_catalog(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("scanner roto")

    monkeypatch.setattr(g, "_scan_allergen_violations", _boom)
    days = [{"meals": [{"name": "Guiso", "ingredients": [],
                        "recipe": ["Agrega el apio y el brocoli picados."]}]}]
    n = g._add_missing_recipe_step_vegetables(days, allergies=["maní"])
    assert n == 0, "con scanner roto y alergias declaradas, NINGÚN candidato debe inyectarse"
    assert days[0]["meals"][0]["ingredients"] == []


# ---------------------------------------------------------------------------
# 2. Parser-based: signature + wiring en las superficies
# ---------------------------------------------------------------------------
def test_signature_has_allergies_param():
    sig = inspect.signature(g._add_missing_recipe_step_vegetables)
    assert "allergies" in sig.parameters
    assert sig.parameters["allergies"].default is None
    sig_fin = inspect.signature(g.finalize_single_meal_recipe_coherence)
    assert "allergies" in sig_fin.parameters
    sig_plan = inspect.signature(g.finalize_plan_data_coherence)
    assert "allergies" in sig_plan.parameters


def test_finalizer_passes_allergies_to_veg_guard():
    assert "_add_missing_recipe_step_vegetables(_wrap, allergies=allergies)" in _GRAPH, \
        "finalize_single_meal_recipe_coherence no pasa allergies al veg-guard"
    assert "_add_missing_recipe_step_vegetables(days, allergies=allergies)" in _GRAPH, \
        "finalize_plan_data_coherence no pasa allergies al veg-guard"


def test_wired_in_swap():
    assert "allergies=allergies)" in _AGENT and "P0-VEG-GUARD-ALLERGEN" in _AGENT, \
        "swap (agent.py) no pasa allergies al finalizer"


def test_wired_in_chat_modify():
    # Sin paréntesis de cierre: P2-CHAT-EXPLICIT-SLOT-WISH añadió `skip_night_rice=` después de
    # `allergies=` en el mismo call → el match exacto con `)` quedó stale. El contrato es que el
    # finalizer reciba las allergies del perfil, no la forma exacta del call.
    assert "allergies=_clin_allergies" in _TOOLS and "P0-VEG-GUARD-ALLERGEN" in _TOOLS, \
        "chat-modify (tools.py) no pasa allergies al finalizer"


def test_wired_in_recalculate_and_expand():
    assert "_fin_rc_rc(_m, allergies=_rc_allergies, portion_floors=False)" in _PLANS, \
        "/recalculate no pasa allergies al finalizer"
    assert "_veg_exp(_wrap_exp, allergies=_expand_allergies)" in _PLANS, \
        "/recipe/expand no pasa allergies al veg-guard"
    assert "P0-VEG-GUARD-ALLERGEN" in _PLANS


def test_marker_anchor_present():
    assert "P0-VEG-GUARD-ALLERGEN" in _GRAPH, "falta el tooltip-anchor en graph_orchestrator.py"
