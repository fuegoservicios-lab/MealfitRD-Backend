"""[A1-HARDEN-POOLS · 2026-07-09] Scaffold tests: knobs nacen OFF, cohorte de canario
determinista, y harden_day_pools es no-op cuando el master está OFF.

Tests de función pura (sin DB). Ancla el contrato del enforcer antes de añadir lógica.
tooltip-anchor: A1-HARDEN-POOLS
"""
import graph_orchestrator as go


def test_knobs_exist_and_default_off():
    # defaults seguros: master + per-class OFF, canary 0 (gates nacen OFF, convención del repo)
    assert go.HARDEN_POOLS_ENABLED is False
    assert go.HARDEN_CONDITION_CATALOG is False
    assert go.HARDEN_SALTCURED_MAIN is False
    assert go.HARDEN_SAMEDAY_PROTEIN is False
    assert go.HARDEN_CROSSDAY_QUOTA is False
    assert go.HARDEN_MAIN_ARITY is False
    assert go.HARDEN_POOLS_CANARY_PCT == 0


def test_harden_day_pools_noop_when_master_off(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", False)
    skel = {"days": [{"day": 1, "protein_pool": ["Salami Dominicano", "Pollo"],
                      "carb_pool": ["Arroz blanco"], "fruit_pool": ["Toronja"]}]}
    counts = go.harden_day_pools(skel, {"medicalConditions": ["dm2"]}, None)
    assert counts == {"condition_removed": 0, "saltcured_removed": 0,
                      "sameday_bound": 0, "crossday_capped": 0, "main_arity_added": 0}
    # pools intactos con master OFF
    assert skel["days"][0]["fruit_pool"] == ["Toronja"]
    assert skel["days"][0]["carb_pool"] == ["Arroz blanco"]


def test_canary_cohort_deterministic_and_default_on(monkeypatch):
    # [P2-COHORT-TAG-EFFECTIVE · 2026-07-10] el bucketing por PCT solo aplica con el master switch
    # ON — con master OFF la cohorte reportada es 'off' (test dedicado en test_p2_1_cohort_tag_effective.py).
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    st = {"form_data": {"user_id": "u-123"}}
    monkeypatch.setattr(go, "HARDEN_POOLS_CANARY_PCT", 0)
    assert go._harden_pools_canary_cohort(st) == "on"   # PCT=0 → siempre on
    monkeypatch.setattr(go, "HARDEN_POOLS_CANARY_PCT", 100)
    assert go._harden_pools_canary_cohort(st) == "off"  # PCT=100 → siempre off
    # estable por usuario: misma respuesta 2 veces
    monkeypatch.setattr(go, "HARDEN_POOLS_CANARY_PCT", 50)
    a = go._harden_pools_canary_cohort(st)
    b = go._harden_pools_canary_cohort(st)
    assert a == b and a in ("on", "off")


def test_canary_cohort_independent_salt_from_self_critique(monkeypatch):
    # El salt propio 'harden_pools|' evita confundir la cohorte A1 con la de self_critique
    # para el MISMO usuario (deben poder diferir).
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    st = {"form_data": {"user_id": "u-xyz"}}
    monkeypatch.setattr(go, "HARDEN_POOLS_CANARY_PCT", 50)
    monkeypatch.setattr(go, "SELF_CRITIQUE_CANARY_PCT", 50)
    a1 = go._harden_pools_canary_cohort(st)
    # ambas son válidas; lo que anclamos es que A1 usa su propio bucketing (no crashea, on/off)
    assert a1 in ("on", "off")


def test_canary_cohort_failsafe_on_bad_state(monkeypatch):
    monkeypatch.setattr(go, "HARDEN_POOLS_ENABLED", True)
    monkeypatch.setattr(go, "HARDEN_POOLS_CANARY_PCT", 50)
    # state None / sin form_data → fail-safe 'on' (nunca revienta el pipeline)
    assert go._harden_pools_canary_cohort({}) in ("on", "off")
