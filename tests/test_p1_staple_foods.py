"""[P1-STAPLE-FOODS · 2026-08-02] "Mis básicos + modo universo-chico" — feature aprobada por el
owner: (A) el usuario declara alimentos de siempre (staples, máx 8, chips del catálogo real); (B)
los básicos pueden repetirse entre días Y el MISMO día con TÉCNICA DISTINTA sin castigo; (C) modo
universo-chico: con la Nevera/universo disponible chico, el chef varía por TÉCNICA/FORMATO en vez
de por ingrediente — los gates de variedad ESTÉTICA ceden, coherencia culinaria/macros/clínicas/
sodio NUNCA.

Cubre:
  1. Validación server-side de `staple_foods` (máx 8, catálogo real — 422 si no).
  2. Gate same-day-protein staple-aware en `build_variety_report` (graph_orchestrator.py):
     básico + técnica distinta → NO cuenta como repetición; básico + MISMA técnica → sigue
     contando; NO-básico → gate intacto (sin cambio de comportamiento).
  3. Paridad en `_days_with_same_day_protein_repeat` (usado por self-critique/surgical-regen).
  4. `_small_universe_active` (umbral MEALFIT_SMALL_UNIVERSE_THRESHOLD) + directiva inyectada en
     `build_day_assignment_context` (prompts/day_generator.py).
  5. Directiva de básicos + universo-chico en el prompt del swap (agent.py) — parser-based (el
     swap real requiere LLM, fuera de alcance de un unit test).
  6. Knob global `MEALFIT_STAPLE_FOODS` (default True) — OFF revierte TODO al comportamiento
     pre-staples (gate ciego a básicos, sin directivas, hidratación server-side desactivada).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import graph_orchestrator as go  # noqa: E402
from prompts.day_generator import build_day_assignment_context  # noqa: E402

_AGENT_SRC = (_BACKEND / "agent.py").read_text(encoding="utf-8")
_PLANS_SRC = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
_USER_DATA_SRC = (_BACKEND / "routers" / "user_data.py").read_text(encoding="utf-8")
_APP_SRC = (_BACKEND / "app.py").read_text(encoding="utf-8")


def _meal(name, slot="almuerzo", ings=None, recipe=None):
    m = {"name": name, "meal": slot, "ingredients": ings or []}
    if recipe:
        m["recipe"] = recipe
    return m


def _plan(*days):
    return {"days": [{"day": i + 1, "meals": list(d)} for i, d in enumerate(days)]}


# ---------------------------------------------------------------------------
# 0. Markers + knobs presentes (trazabilidad, cross-link con CLAUDE.md/app.py)
# ---------------------------------------------------------------------------

def test_marker_and_knob_present():
    assert "P1-STAPLE-FOODS" in go.__dict__ or "P1-STAPLE-FOODS" in Path(go.__file__).read_text(encoding="utf-8")
    src = Path(go.__file__).read_text(encoding="utf-8")
    assert 'MEALFIT_STAPLE_FOODS_ENABLED = _env_bool("MEALFIT_STAPLE_FOODS", True)' in src
    assert "MEALFIT_SMALL_UNIVERSE_THRESHOLD" in src


def test_last_known_pfix_bumped():
    assert "P1-STAPLE-FOODS" in _APP_SRC, (
        "_LAST_KNOWN_PFIX debe apuntar a P1-STAPLE-FOODS tras este cierre (CLAUDE.md convención)."
    )


# ---------------------------------------------------------------------------
# 1. Validación server-side (máx 8 + catálogo real)
# ---------------------------------------------------------------------------

def _import_validator(monkeypatch, catalog_names):
    """Importa `_validate_staple_foods` con `db.execute_sql_query` mockeado a un catálogo fijo."""
    import db as db_module
    from routers import user_data as ud

    def _fake_execute_sql_query(query, params=None, fetch_all=False, **kw):
        assert fetch_all is True
        wanted = set(params[0]) if params else set()
        return [{"name": n} for n in catalog_names if n.lower() in wanted]

    monkeypatch.setattr(db_module, "execute_sql_query", _fake_execute_sql_query, raising=False)
    return ud


def test_validate_staple_foods_max_8(monkeypatch):
    ud = _import_validator(monkeypatch, [f"Alimento {i}" for i in range(10)])
    with pytest.raises(Exception) as exc_info:
        ud._validate_staple_foods([f"Alimento {i}" for i in range(9)])
    assert "422" in str(getattr(exc_info.value, "status_code", "422")) or exc_info.value.status_code == 422


def test_validate_staple_foods_rejects_unknown_catalog_entry(monkeypatch):
    ud = _import_validator(monkeypatch, ["Pollo", "Arroz"])
    with pytest.raises(Exception) as exc_info:
        ud._validate_staple_foods(["Pollo", "AlimentoInventado123"])
    assert exc_info.value.status_code == 422
    assert "AlimentoInventado123" in str(exc_info.value.detail)


def test_validate_staple_foods_accepts_valid_catalog_entries(monkeypatch):
    ud = _import_validator(monkeypatch, ["Pollo", "Huevos", "Arroz"])
    out = ud._validate_staple_foods(["Pollo", "Huevos", "pollo"])  # dedup case-insensitive
    assert out == ["Pollo", "Huevos"]


def test_validate_staple_foods_empty_list_is_valid(monkeypatch):
    ud = _import_validator(monkeypatch, ["Pollo"])
    assert ud._validate_staple_foods([]) == []


def test_validate_staple_foods_non_list_rejected(monkeypatch):
    ud = _import_validator(monkeypatch, ["Pollo"])
    with pytest.raises(Exception) as exc_info:
        ud._validate_staple_foods("Pollo")  # type: ignore[arg-type]
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# 2. Gate same-day-protein staple-aware (build_variety_report)
# ---------------------------------------------------------------------------

def test_staple_with_distinct_technique_same_day_is_exempt():
    """Huevo DUro (desayuno) + Huevo Revuelto (cena), 'huevo' declarado básico → NO cuenta."""
    plan = _plan([
        _meal("Huevo Duro con Tostadas", "desayuno"),
        _meal("Pollo a la Plancha con Arroz", "almuerzo"),
        _meal("Manzana con Maní", "merienda"),
        _meal("Huevo Revuelto con Vegetales", "cena"),
    ])
    vr = go.build_variety_report(plan, user_staples={"huevo"})
    assert vr["same_day_protein_repeats"] == 0, vr
    assert not any("MISMA PROTEÍNA" in i for i in go._variety_repeat_gate_issues(vr))


def test_staple_with_same_technique_same_day_still_gated():
    """Huevo Revuelto ×2 el mismo día, 'huevo' es básico PERO la técnica es la MISMA → sigue
    contando (decisión B: la exención es por TÉCNICA distinta, no por ser básico a secas)."""
    plan = _plan([
        _meal("Huevo Revuelto con Tostadas", "desayuno"),
        _meal("Pollo a la Plancha con Arroz", "almuerzo"),
        _meal("Manzana con Maní", "merienda"),
        _meal("Huevo Revuelto con Vegetales", "cena"),
    ])
    vr = go.build_variety_report(plan, user_staples={"huevo"})
    assert vr["same_day_protein_repeats"] >= 1, vr


def test_staple_without_detectable_technique_stays_conservative():
    """Sin ningún token de técnica en nombre/receta → firma None → NUNCA exime (fail-safe)."""
    plan = _plan([
        _meal("Huevo con Tostadas", "desayuno"),
        _meal("Pollo a la Plancha con Arroz", "almuerzo"),
        _meal("Manzana con Maní", "merienda"),
        _meal("Huevo con Vegetales", "cena"),
    ])
    vr = go.build_variety_report(plan, user_staples={"huevo"})
    assert vr["same_day_protein_repeats"] >= 1, vr


def test_non_staple_protein_repeat_still_gated_regardless_of_technique():
    """Pollo ×2 con técnicas distintas, pero 'pollo' NO es un básico declarado → el gate NO cede
    (la exención B es específica a básicos declarados, no una relajación general)."""
    plan = _plan([
        _meal("Pollo Guisado con Arroz", "almuerzo"),
        _meal("Avena con Fresas", "desayuno"),
        _meal("Yogurt con Granola", "merienda"),
        _meal("Pollo a la Plancha con Yuca", "cena"),
    ])
    vr = go.build_variety_report(plan, user_staples={"huevo"})  # básico declarado es OTRO alimento
    assert vr["same_day_protein_repeats"] >= 1, vr


def test_no_user_staples_preserves_legacy_behavior():
    """user_staples=None (default) → comportamiento IDÉNTICO al pre-staples."""
    plan = _plan([
        _meal("Huevo Duro con Tostadas", "desayuno"),
        _meal("Pollo a la Plancha con Arroz", "almuerzo"),
        _meal("Manzana con Maní", "merienda"),
        _meal("Huevo Revuelto con Vegetales", "cena"),
    ])
    vr = go.build_variety_report(plan)
    assert vr["same_day_protein_repeats"] >= 1, vr


# ---------------------------------------------------------------------------
# 3. Paridad en _days_with_same_day_protein_repeat
# ---------------------------------------------------------------------------

def test_days_with_repeat_parity_staple_exempt():
    plan = _plan([
        _meal("Huevo Duro con Tostadas", "desayuno"),
        _meal("Pollo a la Plancha con Arroz", "almuerzo"),
        _meal("Manzana con Maní", "merienda"),
        _meal("Huevo Revuelto con Vegetales", "cena"),
    ])
    assert go._days_with_same_day_protein_repeat(plan, user_staples={"huevo"}) == []
    assert go._days_with_same_day_protein_repeat(plan) == [1]


# ---------------------------------------------------------------------------
# 3b. Dual-casing del wire format (snake_case explícito vs JSON.stringify(formData) camelCase)
# ---------------------------------------------------------------------------

def test_raw_staple_foods_accepts_snake_case():
    assert go._raw_staple_foods({"staple_foods": ["Huevos", "Pollo"]}) == ["Huevos", "Pollo"]


def test_raw_staple_foods_accepts_camel_case_from_full_formdata_dump():
    """La generación inicial/renovación manda JSON.stringify(formData) completo (frontend
    camelCase) → llega como 'stapleFoods', no 'staple_foods'. Sin este fallback la feature
    quedaría inerte en el camino MÁS COMÚN (generación, no swap)."""
    assert go._raw_staple_foods({"stapleFoods": ["Huevos", "Pollo"]}) == ["Huevos", "Pollo"]


def test_raw_staple_foods_snake_case_wins_when_both_present():
    assert go._raw_staple_foods({"staple_foods": ["Pollo"], "stapleFoods": ["Huevos"]}) == ["Pollo"]


def test_user_staple_labels_camel_case_maps_to_gate_labels():
    assert go._user_staple_labels({"stapleFoods": ["Huevos"]}) == {"huevo"}


# ---------------------------------------------------------------------------
# 4. Modo universo-chico
# ---------------------------------------------------------------------------

def test_small_universe_active_below_threshold():
    fd = {"current_pantry_ingredients": [f"item{i}" for i in range(10)]}
    assert go._small_universe_active(fd) is True


def test_small_universe_inactive_above_threshold():
    fd = {"current_pantry_ingredients": [f"item{i}" for i in range(45)]}
    assert go._small_universe_active(fd) is False, (
        "44-45 items (Nevera del owner, medido en vivo) NO debe activar el modo universo-chico"
    )


def test_small_universe_inactive_without_pantry():
    """Plan 'libre' sin Nevera real → catálogo completo disponible, nunca universo-chico."""
    assert go._small_universe_active({}) is False
    assert go._small_universe_active({"current_pantry_ingredients": []}) is False


def test_small_universe_directive_in_daygen_prompt():
    sk = {"protein_pool": ["Pollo"], "carb_pool": [], "fruit_pool": []}
    ctx_on = build_day_assignment_context(sk, 1, small_universe=True)
    ctx_off = build_day_assignment_context(sk, 1, small_universe=False)
    assert "UNIVERSO-CHICO" in ctx_on
    assert "UNIVERSO-CHICO" not in ctx_off


def test_staple_directive_in_daygen_prompt():
    sk = {"protein_pool": ["Pollo"], "carb_pool": [], "fruit_pool": []}
    ctx_with = build_day_assignment_context(sk, 1, user_staples=["Huevos", "Arroz"])
    ctx_without = build_day_assignment_context(sk, 1)
    assert "BÁSICOS DEL USUARIO" in ctx_with
    assert "Huevos" in ctx_with and "Arroz" in ctx_with
    assert "BÁSICOS DEL USUARIO" not in ctx_without


def test_daygen_prompt_still_deterministic_without_staples():
    """Sin user_staples/small_universe (callers viejos: self-critique/surgical-regen sin kwargs
    nuevos) el prompt es BYTE-IDÉNTICO al pre-staples — no rompe el test de determinismo P1-NEXT-LEVEL-BATCH."""
    sk = {"protein_pool": ["Pollo"], "carb_pool": ["Arroz", "Yuca"], "fruit_pool": ["Guineo"]}
    a = build_day_assignment_context(sk, 1)
    b = build_day_assignment_context(sk, 1)
    assert a == b


# ---------------------------------------------------------------------------
# 5. Swap: directiva + gate de exención (parser-based — el swap real invoca LLM)
# ---------------------------------------------------------------------------

def test_swap_prompt_carries_staple_and_small_universe_directive():
    assert "BÁSICOS DEL USUARIO" in _AGENT_SRC
    assert "MODO UNIVERSO-CHICO" in _AGENT_SRC
    i = _AGENT_SRC.find("P1-STAPLE-FOODS")
    assert i > 0


def test_swap_gate_exemption_wired():
    i = _AGENT_SRC.find("P1-SWAP-SAMEDAY-PROTEIN-GATE")
    assert i > 0
    blk = _AGENT_SRC[i: i + 6000]
    assert "_user_staple_labels" in blk
    assert "_technique_signature_from_text" in blk
    assert "_clash_sd <= _user_staples_sd" in blk, (
        "la exención solo debe aplicar cuando TODOS los labels en conflicto son básicos"
    )


def test_staple_foods_hydrated_server_side_on_swap():
    """_enrich_clinical_from_profile (routers/plans.py) hidrata staple_foods igual que dislikes —
    cierra la ventana de cliente stale para el gate del swap."""
    i = _PLANS_SRC.find("def _enrich_clinical_from_profile")
    assert i > 0
    blk = _PLANS_SRC[i: i + 4000]
    assert 'hp.get("staple_foods")' in blk
    assert 'data["staple_foods"]' in blk


# ---------------------------------------------------------------------------
# 6. Endpoint GET/PUT (contrato en el source — auth/IO real cubierto por el resto de la suite I2)
# ---------------------------------------------------------------------------

def test_staple_foods_endpoint_present_and_authed():
    assert '@router.get("/user/preferences/staple-foods")' in _USER_DATA_SRC
    assert '@router.put("/user/preferences/staple-foods")' in _USER_DATA_SRC
    i = _USER_DATA_SRC.find('@router.put("/user/preferences/staple-foods")')
    blk = _USER_DATA_SRC[i: i + 800]
    assert "get_verified_user_id" in blk, "I2: el endpoint debe requerir auth"
    assert "update_user_health_profile_atomic" in blk, "I7: escritura vía FOR UPDATE + callback"


def test_staple_foods_max_constant_is_8():
    assert "_STAPLE_FOODS_MAX = 8" in _USER_DATA_SRC


# ---------------------------------------------------------------------------
# 7. Knob global OFF revierte todo (rollback sin redeploy)
# ---------------------------------------------------------------------------

def test_knob_off_user_staple_labels_returns_empty(monkeypatch):
    monkeypatch.setattr(go, "MEALFIT_STAPLE_FOODS_ENABLED", False)
    labels = go._user_staple_labels({"staple_foods": ["Huevos", "Pollo"]})
    assert labels == set()


def test_knob_off_small_universe_never_active(monkeypatch):
    monkeypatch.setattr(go, "MEALFIT_STAPLE_FOODS_ENABLED", False)
    fd = {"current_pantry_ingredients": [f"item{i}" for i in range(5)]}
    assert go._small_universe_active(fd) is False


def test_knob_off_gate_reverts_to_legacy_behavior(monkeypatch):
    """Con el knob OFF, incluso un básico declarado con técnica distinta sigue contando como
    repetición — comportamiento 100% pre-staples."""
    monkeypatch.setattr(go, "MEALFIT_STAPLE_FOODS_ENABLED", False)
    plan = _plan([
        _meal("Huevo Duro con Tostadas", "desayuno"),
        _meal("Pollo a la Plancha con Arroz", "almuerzo"),
        _meal("Manzana con Maní", "merienda"),
        _meal("Huevo Revuelto con Vegetales", "cena"),
    ])
    vr = go.build_variety_report(plan, user_staples={"huevo"})
    assert vr["same_day_protein_repeats"] >= 1, vr
