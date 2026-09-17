# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-76 · 2026-09-17] Dos peticiones del dueño la noche del 16-17:

1. «Prefiero un bloqueo total hasta medianoche»: el botón «Nuevo chat» va deshabilitado mientras el chat abierto es
   el de hoy (frontend; aquí, el contrato entre repos).
2. «No debe preguntarme si me faltó registrar la cena, ya lo debería saber y ser más proactivo»: tras registrar,
   `log_consumed_meal` dice qué comidas de ESE día siguen sin registrar, el bloque del diario de días anteriores lo
   anota, y el prompt exige ofrecer LA QUE FALTA por su nombre o no preguntar.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
_FRONT = _BACKEND.parent / "frontend"


def _src(rel: str) -> str:
    return (_BACKEND / rel).read_text(encoding="utf-8")


def _front(rel: str) -> str:
    p = _FRONT / rel
    if not p.exists():
        pytest.skip("sin el repo del frontend al lado")
    return p.read_text(encoding="utf-8")


# ── SSOT: qué comidas faltan ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("valor,slot", [
    ("Desayuno", "desayuno"), ("desayuno", "desayuno"), ("ALMUERZO", "almuerzo"), ("Comida", "almuerzo"),
    ("Cena", "cena"), ("Merienda", "merienda"), ("snack", "merienda"), ("Colación", "merienda"),
    ("Merienda PM", "merienda"), ("", None), (None, None), ("postre", None),
])
def test_slot_canonico(valor, slot):
    from chat_history_context import slot_canonico
    assert slot_canonico(valor) == slot


def test_las_tres_principales_siempre_y_la_merienda_solo_si_el_plan_la_trae():
    from chat_history_context import comidas_sin_registrar
    rows = [{"meal_type": "desayuno"}, {"meal_type": "almuerzo"}]
    assert comidas_sin_registrar(rows) == ["cena"]
    assert comidas_sin_registrar(rows, {"meals": [{"meal": "Desayuno"}, {"meal": "Merienda"}, {"meal": "Cena"}]}) == ["merienda", "cena"]
    # el registro de resolve_day_dates envuelve el día en `day`
    assert comidas_sin_registrar(rows, {"date": date(2026, 9, 16), "day": {"meals": [{"meal": "Snack"}]}}) == ["merienda", "cena"]
    assert comidas_sin_registrar(rows + [{"meal_type": "cena"}]) == []
    assert comidas_sin_registrar([]) == ["desayuno", "almuerzo", "cena"]
    assert comidas_sin_registrar([{"meal_type": "snack"}], {"meals": [{"meal": "Merienda"}]}) == ["desayuno", "almuerzo", "cena"]


# ── El bloque del diario de días anteriores lo anota ──────────────────────────────────────────────

def test_el_bloque_del_diario_dice_que_falta_por_dia():
    from chat_history_context import build_past_diary_block
    hoy = date(2026, 9, 17)
    rows = [
        {"meal_name": "Huevos", "meal_type": "desayuno", "calories": 300, "consumed_at": "2026-09-16T16:58:00+00:00"},
        {"meal_name": "Arroz con res", "meal_type": "almuerzo", "calories": 850, "consumed_at": "2026-09-16T17:30:00+00:00"},
    ]
    plan = {"days": [{"day": 1, "meals": [{"meal": "Desayuno", "name": "a"}, {"meal": "Merienda", "name": "b"}, {"meal": "Cena", "name": "c"}]}],
            "_plan_start_date": "2026-09-16", "grocery_start_date": "2026-09-16"}
    bloque = build_past_diary_block(rows, hoy, days_back=2, plan_data=plan)
    linea_ayer = [l for l in bloque.splitlines() if "16" in l and "Huevos" in l][0]
    assert linea_ayer.endswith("sin registrar: merienda, cena")
    assert "SIN REGISTRO" in bloque                      # el día 15 sigue declarado vacío
    assert "«sin registrar: …»" in bloque                # el pie lo explica
    sin_plan = build_past_diary_block(rows, hoy, days_back=2)
    linea_sin_plan = [l for l in sin_plan.splitlines() if "Huevos" in l][0]
    assert linea_sin_plan.endswith("sin registrar: cena")   # sin plan del día, la merienda no cuenta


def test_el_agente_pasa_el_plan_al_bloque():
    assert "plan_data=current_plan)  # [P1-PLAN-LOTE-76]" in _src("agent.py")


# ── La herramienta lo dice tras registrar ─────────────────────────────────────────────────────────

@pytest.fixture
def herramienta(monkeypatch):
    import tools
    import db_facts
    monkeypatch.setattr(tools, "db_log_consumed_meal", lambda *a, **k: "meal-1")
    monkeypatch.setattr(tools, "user_tz_offset_min", lambda uid: 240)
    monkeypatch.setattr(tools, "_rescue_dinner_slot", lambda uid, mt, cal, d: mt)
    import db
    monkeypatch.setattr(db, "execute_sql_query", lambda *a, **k: None)          # dup-guard: nada registrado
    monkeypatch.setattr(tools, "get_latest_usable_meal_plan", lambda uid: {"days": []})
    import db_inventory
    monkeypatch.setattr(db_inventory, "deduct_consumed_meal_from_inventory", lambda *a, **k: None)
    estado = {"rows": []}
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", lambda uid, date_str=None, tz_offset_mins=None: estado["rows"])
    return tools, estado


def test_tras_registrar_dice_que_falta_y_pide_ofrecerla_por_su_nombre(herramienta):
    tools, estado = herramienta
    estado["rows"] = [{"meal_type": "desayuno"}, {"meal_type": "almuerzo"}, {"meal_type": "cena"}]
    out = tools.log_consumed_meal.func("u1", "Huevos hervidos", 220, 9, 2, 14, meal_type="cena", days_ago=1) \
        if hasattr(tools.log_consumed_meal, "func") else tools.log_consumed_meal("u1", "Huevos hervidos", 220, 9, 2, 14, meal_type="cena", days_ago=1)
    assert "¡Éxito!" in out
    assert "ayer ya tiene todas sus comidas registradas — NO preguntes" in out
    estado["rows"] = [{"meal_type": "desayuno"}, {"meal_type": "cena"}]
    out = tools.log_consumed_meal.func("u1", "Huevos", 220, 9, 2, 14, meal_type="cena", days_ago=1) \
        if hasattr(tools.log_consumed_meal, "func") else tools.log_consumed_meal("u1", "Huevos", 220, 9, 2, 14, meal_type="cena", days_ago=1)
    assert "ayer sigue sin registrar almuerzo — cierra ofreciendo agregar LA QUE FALTA por su nombre" in out


def test_el_calculo_es_best_effort(herramienta, monkeypatch):
    tools, estado = herramienta
    import db_facts
    def revienta(*a, **k):
        raise RuntimeError("bd caída")
    monkeypatch.setattr(db_facts, "get_consumed_meals_today", revienta)
    fn = tools.log_consumed_meal.func if hasattr(tools.log_consumed_meal, "func") else tools.log_consumed_meal
    out = fn("u1", "Huevos", 220, 9, 2, 14, meal_type="cena", days_ago=0)
    assert "¡Éxito!" in out and "Para el asistente" not in out


def test_el_prompt_prohibe_la_pregunta_generica():
    p = _src("prompts/chat_agent.py")
    assert "[P1-PLAN-LOTE-76] LA COMIDA QUE FALTA, POR SU NOMBRE" in p
    assert "Prohibida la pregunta genérica '¿Te falta algo más por registrar?'" in p


# ── Frontend: el botón bloqueado ──────────────────────────────────────────────────────────────────

def test_el_boton_nuevo_chat_va_bloqueado_mientras_el_chat_es_el_de_hoy():
    regla = _front("src/utils/chatSessionDay.js")
    assert "export const nuevoChatBloqueado = (sessionId, hoy = hoyLocal()) => diaAnotadoDe(sessionId) === hoy;" in regla
    barra = _front("src/components/agent/SidebarRecientes.jsx")
    assert "const bloqueado = nuevoChatBloqueado(currentSessionId);" in barra
    assert "disabled={bloqueado}" in barra and "aria-disabled={bloqueado}" in barra
    assert "padding: '0.75rem 1rem 0.5rem'" in barra           # la cabecera conserva sus 84 px
    agente = _front("src/pages/AgentPage.jsx")
    assert "if (nuevoChatBloqueado(currentSessionIdRef.current)) return;" in agente
    prueba = _front("src/__tests__/Agent.renovacion_diaria.test.jsx")
    assert "el botón «Nuevo chat» está bloqueado mientras el chat abierto es el de hoy" in prueba


def test_marcador_y_documentos():
    assert 'P1-PLAN-LOTE-76 · 2026-09-17' in _src("app.py")
    assert "P1-PLAN-LOTE-76" in _src("docs/chat_sesion_del_dia.md")
    assert "comidas_sin_registrar" in _src("docs/chat_past_days_memory.md")
