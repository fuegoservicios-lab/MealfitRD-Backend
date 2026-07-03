"""[P1-DAYGEN-RETRY-FLASH-NET · 2026-07-03] Flash como red del retry del day-gen.

Residuo del gym baseline (eje entrega): 2/20 planes cayeron a fallback MATEMÁTICO total —
uno de ellos maintenance SIN condiciones. Minado del log: el circuit breaker de
deepseek-v4-pro estuvo abierto (172 menciones) y el chain de retry del day-gen era
`[deepseek-v4-pro]` A SECAS → "Circuit Breaker OPEN para todo el chain" → todos los
workers muertos → plan de contingencia. Un día real generado por flash (validado por los
mismos gates de review) es estrictamente mejor que un día matemático.

Cierra: retry chain = [pro, flash] (pro primero — calidad intacta con breaker sano; el
cascade solo cae a flash con pro caído/abierto). Bariátrico intacto ([pro] deliberado:
su fallback matemático está curado clínicamente).
"""
from __future__ import annotations

import re
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
_GO = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")

_NON = {"medicalConditions": ["Ninguna"]}
_BAR = {"medicalConditions": ["Cirugía bariátrica"]}


def test_marker_bumped():
    src = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"', src)
    assert m, "falta _LAST_KNOWN_PFIX"
    if "P1-DAYGEN-RETRY-FLASH-NET" in m.group(1):
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", m.group(1))
    assert fecha and fecha.group(1) >= "2026-07-03"


def test_retry_chain_has_flash_net(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_FLASH_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "deepseek-v4-pro")
    monkeypatch.setattr(g, "DAY_GEN_RETRY_USE_PRO", True)
    chain = g._day_model_chain(_NON, 2)
    assert chain == ["deepseek-v4-pro", "deepseek-v4-flash"], \
        "el retry debe llevar flash como red — [pro] a secas + breaker abierto = fallback matemático"
    # attempt 3 igual (todo retry lleva la red)
    assert g._day_model_chain(_NON, 3) == ["deepseek-v4-pro", "deepseek-v4-flash"]


def test_bariatric_stays_pro_only(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_FLASH_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "deepseek-v4-pro")
    if not g.BARIATRIC_DAYGEN_PRO:
        return  # knob off en el baseline de tests → decisión cubierta por su propio test
    assert g._day_model_chain(_BAR, 2) == ["deepseek-v4-pro"], \
        "bariátrico NO degrada a flash (decisión clínica deliberada; su fallback está curado)"


def test_attempt1_unchanged(monkeypatch):
    import graph_orchestrator as g
    monkeypatch.setattr(g, "_FLASH_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setattr(g, "_PRO_MODEL_NAME", "deepseek-v4-pro")
    assert g._day_model_chain(_NON, 1) == ["deepseek-v4-flash", "deepseek-v4-pro"]
