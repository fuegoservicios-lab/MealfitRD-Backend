"""[P1-PLAN-LOTE-982 · 2026-11-11] Una comida principal en dos tandas cercanas es UNA comida.

Caso vivo del dueño: 3 panes con pollo a las 20:15 («Me comí tres pan de agua…») y, 20 min después, la foto del
mangú («Esta es la cena, había otro salami…»): quedaron DOS filas «Cena». Lo comió todo, en dos tandas: dentro del
TTL de fusión (30 min) la segunda se suma a la primera aunque no diga «fue parte de la cena».
"""
from __future__ import annotations

import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import db_facts  # noqa: E402
from agent import _intentar_fusionar_alimento_en_comida_existente, _slot_principal_dentro_de_ttl  # noqa: E402


def _panes(minutos):
    now = datetime.now(timezone.utc)
    return [{
        "id": "cena-panes", "meal_name": "3 panes de agua con ensalada de pollo y mayonesa", "meal_type": "cena",
        "calories": 920, "protein": 51, "carbs": 105, "healthy_fats": 34,
        "ingredients": ["pan de agua (3 unidades, ≈180 g)", "pollo cocido desmenuzado (6 cucharadas, ≈90 g)"],
        "created_at": (now - timedelta(minutes=minutos)).isoformat(),
        "consumed_at": (now - timedelta(minutes=minutos)).isoformat(),
    }]


def _mensajes(texto="Esta es la cena, había otro salami y es un solo huevo"):
    return [
        HumanMessage(content="Me comí tres pan de agua con pollo de ensalada, 6 cucharadas de pollo + ensalada y mayonesa"),
        AIMessage(content="", tool_calls=[{"name": "log_consumed_meal", "args": {"meal_type": "cena"}, "id": "t1"}]),
        ToolMessage(content="¡Éxito! [ID_REGISTRO_DIARIO: cena-panes]", tool_call_id="t1"),
        AIMessage(content="Anoté tu cena…"),
        HumanMessage(content=texto),
    ]


_MANGU = {"meal_name": "Mangú de auyama con 1 huevo frito, salami frito y queso frito", "calories": 640,
          "protein": 27, "carbs": 40, "healthy_fats": 46, "meal_type": "cena",
          "ingredients": ["mangú de auyama (≈200 g)", "1 huevo frito", "3 lascas de salami frito", "1 lasca de queso frito"]}


def test_caso_del_dueno_se_fusiona_en_la_cena_anterior(monkeypatch):
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda *a, **k: _panes(20))
    fused = _intentar_fusionar_alimento_en_comida_existente("u", dict(_MANGU), _mensajes())
    assert fused is not None
    assert fused["meal_id"] == "cena-panes"
    assert fused["calories"] == 920 + 640


def test_fuera_del_ttl_sigue_siendo_aparte(monkeypatch):
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda *a, **k: _panes(45))
    assert _slot_principal_dentro_de_ttl("u", dict(_MANGU), _mensajes()) is None


def test_otra_franja_no_se_fusiona(monkeypatch):
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda *a, **k: _panes(10))
    assert _slot_principal_dentro_de_ttl("u", {**_MANGU, "meal_type": "merienda"}, _mensajes()) is None


def test_repeticion_declarada_tiene_su_camino(monkeypatch):
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda *a, **k: _panes(10))
    assert _slot_principal_dentro_de_ttl("u", dict(_MANGU), _mensajes("Repetí el plato")) is None


def test_dias_pasados_y_knob(monkeypatch):
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda *a, **k: _panes(10))
    assert _slot_principal_dentro_de_ttl("u", {**_MANGU, "days_ago": 1}, _mensajes()) is None
    monkeypatch.setenv("MEALFIT_SAME_SLOT_TTL_FUSION", "0")
    assert _slot_principal_dentro_de_ttl("u", dict(_MANGU), _mensajes()) is None


def test_el_guard_de_foto_no_frena_esta_fusion():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "_fusion_misma_franja = True" in src
    assert 'tool_name == "correct_consumed_meal" and not _fusion_misma_franja' in src


def test_marker():
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · ', (_BACKEND / "app.py").read_text(encoding="utf-8"))
    assert m and int(m.group(1)) >= 982
