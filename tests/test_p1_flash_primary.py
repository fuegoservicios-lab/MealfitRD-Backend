"""[P1-FLASH-PRIMARY · 2026-07-31] Flash es el modelo PRIMARIO de todas las
superficies; pro queda SOLO como red post-fallo.

Contexto: el owner midió que `deepseek-v4-flash` es actualmente MEJOR que
`deepseek-v4-pro` (los providers actualizan modelos bajo el mismo ID — la
premisa "pro > flash" de P0-DEEPSEEK-MIGRATION 2026-06-12 caducó). Antes de
este fix el .env local YA forzaba flash en ambos tiers (decisión owner
2026-07-04) pero prod corría el default de código: pro para tiers pagados y
para el reviewer clínico risk-tier.

Contratos que ancla:
  A. Router por tier (llm_provider): TODOS los tiers → flash por default;
     `MEALFIT_MODEL_PAID_TIER` sigue permitiendo divergir (rollback).
  B. Reviewer/fact-checker risk-tier: `_REVIEWER_RISK_TIER_DEFAULT = FLASH`
     (mantener pro habría degradado el gate clínico a propósito bajo la
     premisa nueva).
  C. `_route_model` (pipeline): la rama pagada resuelve vía
     `resolve_model_for_tier` (SSOT llm_provider), no vía `_PRO_MODEL_NAME`.
  D. Cadena del day-gen: flash-PRIMERO en attempt 1, retries y bariátrico;
     pro presente como RED (2º) — nunca primero, nunca ausente.
  E. La red pro NO se colapsa: `_plan_pro_model_name` default sigue siendo
     `DEEPSEEK_PRO` (su valor es ser un modelo DISTINTO con breaker
     independiente — colapsarlo a flash haría no-op los fallbacks de
     P1-DAYGEN-RETRY-FLASH-NET y P1-PLANNER-PRO-FALLBACK).
  F. Marker bumpeado en app.py.
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GO_SRC = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
_LP_SRC = (_BACKEND / "llm_provider.py").read_text(encoding="utf-8")
_APP_SRC = (_BACKEND / "app.py").read_text(encoding="utf-8")


# ------------------------------------------------------------------
# A. Router por tier — defaults del CÓDIGO (env-independiente)
# ------------------------------------------------------------------

def test_a_paid_tier_default_is_flash(monkeypatch):
    import db  # noqa: F401 — fuerza load_dotenv ANTES del delenv (patrón test_b2)
    from llm_provider import DEEPSEEK_FLASH, DEEPSEEK_PRO, model_paid_tier, resolve_model_for_tier

    monkeypatch.delenv("MEALFIT_MODEL_PAID_TIER", raising=False)
    monkeypatch.delenv("MEALFIT_MODEL_FREE_TIER", raising=False)

    assert model_paid_tier() == DEEPSEEK_FLASH
    for tier in ("basic", "plus", "ultra", "gratis", None, "desconocido"):
        assert resolve_model_for_tier(tier) == DEEPSEEK_FLASH

    # Rollback vivo: el knob sigue pudiendo divergir pagado→pro sin redeploy.
    monkeypatch.setenv("MEALFIT_MODEL_PAID_TIER", DEEPSEEK_PRO)
    assert resolve_model_for_tier("ultra") == DEEPSEEK_PRO
    assert resolve_model_for_tier("gratis") == DEEPSEEK_FLASH


# ------------------------------------------------------------------
# B. Risk-tier clínico = flash
# ------------------------------------------------------------------

def test_b_reviewer_risk_tier_default_is_flash():
    assert "_REVIEWER_RISK_TIER_DEFAULT = DEEPSEEK_FLASH" in _GO_SRC, (
        "P1-FLASH-PRIMARY: el risk-tier clínico debe defaultear a flash "
        "(el owner midió flash > pro; pro aquí sería degradar el gate a propósito)"
    )
    assert "_REVIEWER_RISK_TIER_DEFAULT = DEEPSEEK_PRO" not in _GO_SRC


# ------------------------------------------------------------------
# C. _route_model rama pagada vía SSOT llm_provider
# ------------------------------------------------------------------

def test_c_route_model_paid_branch_uses_ssot():
    m = re.search(
        r"def _route_model\(.*?\n(?:.*\n)*?    if tier in PAID_TIERS:\n(?P<body>(?:.{0,200}\n){0,8}?)\s*return _paid_model",
        _GO_SRC,
    )
    assert m, "no encontré la rama pagada de _route_model retornando _paid_model"
    assert "resolve_model_for_tier(tier)" in m.group("body"), (
        "la rama pagada debe resolver vía resolve_model_for_tier (SSOT llm_provider)"
    )
    # La rama pagada ya no retorna la constante de la red pro.
    assert "return _PRO_MODEL_NAME" not in m.group("body")


# ------------------------------------------------------------------
# D. Cadena del day-gen flash-primero (attempt 1, retry y bariátrico)
# ------------------------------------------------------------------

def test_d_day_chain_flash_first(monkeypatch):
    import graph_orchestrator as g

    monkeypatch.setattr(g, "_FLASH_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "deepseek-v4-pro")

    _non = {"medicalConditions": ["Ninguna"]}
    _bar = {"medicalConditions": ["Cirugía bariátrica"]}

    assert g._day_model_chain(_non, 1) == ["deepseek-v4-flash", "deepseek-v4-pro"]
    assert g._day_model_chain(_non, 2) == ["deepseek-v4-flash", "deepseek-v4-pro"]
    if g.BARIATRIC_DAYGEN_PRO:
        assert g._day_model_chain(_bar, 1) == ["deepseek-v4-flash", "deepseek-v4-pro"]

    # El literal invertido (pro primero) no debe reaparecer en el source.
    assert "[_PRO_MODEL_NAME, _FLASH_MODEL_NAME]" not in _GO_SRC


def test_d2_bariatric_model_knob_exists():
    # Rollback per-feature del day-gen bariátrico (P3-PREVIEW-MODEL-KNOB).
    assert 'MEALFIT_BARIATRIC_DAYGEN_MODEL' in _GO_SRC


# ------------------------------------------------------------------
# E. La red pro sobrevive (no colapsar los fallbacks)
# ------------------------------------------------------------------

def test_e_pro_net_not_collapsed():
    m = re.search(
        r"def _plan_pro_model_name\(\).*?return _env_str\(\"MEALFIT_PRO_MODEL\", (\w+)\)",
        _GO_SRC,
        re.DOTALL,
    )
    assert m, "no encontré _plan_pro_model_name"
    assert m.group(1) == "DEEPSEEK_PRO", (
        "la RED post-fallo debe seguir siendo un modelo DISTINTO (breaker "
        "independiente); colapsarla a flash haría no-op los fallbacks de "
        "P1-DAYGEN-RETRY-FLASH-NET / P1-PLANNER-PRO-FALLBACK"
    )
    # Ambos extremos de la cadena presentes: flash primario + pro de red.
    assert "[_FLASH_MODEL_NAME, _PRO_MODEL_NAME]" in _GO_SRC


# ------------------------------------------------------------------
# F. Marker + anchors
# ------------------------------------------------------------------

def test_f_marker_and_anchors():
    assert "P1-FLASH-PRIMARY" in _LP_SRC
    assert "P1-FLASH-PRIMARY" in _GO_SRC
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', _APP_SRC)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-FLASH-PRIMARY" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-31", (
        "el marker debe ser P1-FLASH-PRIMARY o uno posterior"
    )
