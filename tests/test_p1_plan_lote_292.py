# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-292 · 2026-09-25] El formulario separa «¿qué tomas?» de «¿te recomendamos?», lo que toma va a la
Alacena, y la tarjeta del dashboard desaparece. Spec §4. Tooltip-anchor: P1-PLAN-LOTE-292"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(rel):
    return (_BACKEND / rel).read_text(encoding="utf-8")


# ── Task 11: normalizar y el generador ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("fd,esperado", [
    ({"currentSupplements": ["creatine"], "recommendSupplements": True}, {"toma": ["creatine"], "recomendar": True}),
    ({"includeSupplements": True, "selectedSupplements": ["whey_protein"]}, {"toma": ["whey_protein"], "recomendar": False}),
    ({"includeSupplements": True, "selectedSupplements": []}, {"toma": [], "recomendar": True}),
    ({"includeSupplements": False}, {"toma": [], "recomendar": False}),
    ({}, {"toma": [], "recomendar": False}),
])
def test_normalizar(fd, esperado):
    import suplementos
    assert suplementos.normalizar_suplementos(fd) == esperado


def test_el_prompt_incluye_lo_que_toma_y_nunca_recomienda_quemadores():
    from prompts.plan_generator import build_supplements_context
    txt = build_supplements_context({"currentSupplements": ["creatine"], "recommendSupplements": True})
    assert "Creatina" in txt and "NUNCA recomiendes" in txt
    for prohibido in ("Quemador", "Pre-Entreno", "BCAA"):
        assert prohibido in txt.split("NUNCA recomiendes", 1)[1]
    solo = build_supplements_context({"currentSupplements": ["fat_burner"], "recommendSupplements": False})
    assert "Quemador" in solo and "lo toma el usuario" in solo.lower()


def test_nada_que_tomar_ni_recomendar_prohibe_suplementos():
    from prompts.plan_generator import build_supplements_context
    viejo = build_supplements_context({"includeSupplements": False})
    nuevo = build_supplements_context({"currentSupplements": [], "recommendSupplements": False})
    assert nuevo == viejo and viejo


def test_suplementos_dia_con_el_formulario_nuevo():
    import suplementos_dia as sd
    fd = {"currentSupplements": ["creatine"], "recommendSupplements": True}
    assert sd.elegidos(fd) == ["creatine"]
    plan = {"days": [{"supplements": [{"name": "Omega-3 (Aceite de Pescado)", "dose": "1 g", "timing": "Almuerzo", "reason": "x"},
                                      {"name": "Quemador de Grasa Termogénico", "dose": "1", "timing": "x", "reason": "x"}]}]}
    sd.completar(plan, fd)
    nombres = [s["name"] for s in plan["days"][0]["supplements"]]
    assert any("Creatina" in n for n in nombres)          # lo suyo se asegura
    assert any("Omega" in n for n in nombres)             # una recomendación con respaldo se queda
    assert not any("Quemador" in n for n in nombres)      # nunca un quemador recomendado


def test_suplementos_dia_toma_sin_recomendar_es_ni_mas_ni_menos():
    import suplementos_dia as sd
    plan = {"days": [{"supplements": [{"name": "Omega-3 (Aceite de Pescado)", "dose": "1 g", "timing": "x", "reason": "x"}]}]}
    sd.completar(plan, {"currentSupplements": ["creatine"], "recommendSupplements": False})
    assert [s["name"] for s in plan["days"][0]["supplements"]] == ["Creatina Monohidrato"]


def test_el_orquestador_y_el_coach_leen_el_formulario_nuevo():
    go = _src("graph_orchestrator.py")
    assert 'if not form_data.get("includeSupplements"):' not in go
    assert "suplementos_activos(form_data)" in go
    ag = _src("agent.py")
    assert 'form_data.get("includeSupplements")' not in ag and ag.count("normalizar_suplementos(form_data)") == 2


def test_el_router_valida_lo_que_toma():
    from routers import plans
    src = _src("routers/plans.py")
    assert '"currentSupplements"' in src


# ── Task 12: lo que toma, a la Alacena ──────────────────────────────────────────────────────────────────────────────

def test_endpoint_guarda_los_que_toma_y_enciende_la_nevera(monkeypatch):
    import suplementos
    from routers import user_data
    guardados = []
    monkeypatch.setattr(suplementos, "buscar", lambda uid, n: {"id": 1} if n == "Creatina Monohidrato" else None)
    monkeypatch.setattr(suplementos, "guardar", lambda *a, **k: guardados.append((a, k)) or {"ok": True})
    r = user_data.guardar_suplementos_del_formulario(
        user_data.SuplementosFormulario(claves=["creatine", "omega3", "no_existe", "omega3"]), user_id="u")
    assert r == {"guardados": 1}                      # creatina ya estaba; «no_existe» fuera; omega3 una vez
    a, k = guardados[0]
    assert a[1] == "Omega-3 (Aceite de Pescado)" and k["forzar_nevera"] is True and k["usar_estimado"] is False
