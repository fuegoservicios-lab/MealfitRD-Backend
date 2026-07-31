"""[P1-DAYGEN-RETRY-FLASH-NET · 2026-07-03] Flash como red del retry del day-gen.

Residuo del gym baseline (eje entrega): 2/20 planes cayeron a fallback MATEMÁTICO total —
uno de ellos maintenance SIN condiciones. Minado del log: el circuit breaker de
deepseek-v4-pro estuvo abierto (172 menciones) y el chain de retry del day-gen era
`[deepseek-v4-pro]` A SECAS → "Circuit Breaker OPEN para todo el chain" → todos los
workers muertos → plan de contingencia. Un día real generado por flash (validado por los
mismos gates de review) es estrictamente mejor que un día matemático.

Cierra: retry chain con AMBOS modelos (la invariante real: un solo breaker abierto jamás
mata todos los workers).

[P1-FLASH-PRIMARY · 2026-07-31] El ORDEN se invirtió: el owner midió que flash es
actualmente MEJOR que pro → retry = [flash, pro] (flash primario con la directiva
correctiva; pro queda de RED de diversidad). Bariátrico también: [flash, pro] (era [pro]
a secas bajo la premisa vieja). La invariante de ESTE test (dos modelos en la cadena,
breakers independientes) sobrevive intacta con los roles invertidos.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")

_NON = {"medicalConditions": ["Ninguna"]}
_BAR = {"medicalConditions": ["Cirugía bariátrica"]}


@pytest.fixture(autouse=True)
def _sin_tier_routing(monkeypatch):
    """[P1-DAYGEN-TIER-MODEL · 2026-07-31] Estos tests anclan la cadena BASE
    [flash, red]. El routing por tier (Luna primario) tiene su propio ancla en
    test_p1_daygen_tier_model.py; aquí se neutraliza para que las aserciones
    midan el contrato base y no el tier/API-key del entorno de quien corre."""
    import graph_orchestrator as _go
    monkeypatch.setattr(_go, "_daygen_tier_profile", lambda: (None, ""))


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-DAYGEN-RETRY-FLASH-NET" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-03"


def test_retry_chain_has_two_model_net(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_FLASH_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "deepseek-v4-pro")
    monkeypatch.setattr(g, "DAY_GEN_RETRY_USE_PRO", True)
    chain = g._day_model_chain(_NON, 2)
    assert chain == ["deepseek-v4-flash", "deepseek-v4-pro"], \
        "el retry debe llevar DOS modelos (breakers independientes) — flash primario por P1-FLASH-PRIMARY"
    # attempt 3 igual (todo retry lleva la red)
    assert g._day_model_chain(_NON, 3) == ["deepseek-v4-flash", "deepseek-v4-pro"]


def test_bariatric_has_pro_net(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_FLASH_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "deepseek-v4-pro")
    if not g.BARIATRIC_DAYGEN_PRO:
        return  # knob off en el baseline de tests → decisión cubierta por su propio test
    assert g._day_model_chain(_BAR, 2) == ["deepseek-v4-flash", "deepseek-v4-pro"], \
        "bariátrico: flash primario (P1-FLASH-PRIMARY) + pro de red — nunca un solo modelo sin red"


def test_attempt1_unchanged(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_FLASH_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "deepseek-v4-pro")
    assert g._day_model_chain(_NON, 1) == ["deepseek-v4-flash", "deepseek-v4-pro"]
