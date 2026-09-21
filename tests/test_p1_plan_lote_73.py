# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-73 · 2026-09-16] El chat del día se renueva solo también con la pestaña abierta, y una cuenta
regresiva bajo «Nuevo chat» lo anuncia. Más el verbo del aviso de comida («¿Ya cenaste tu merienda?»).

El dueño: «que no se tenga que dar a nuevo chat ni siquiera, que lo haga automático diario, y que lo diga una cuenta
regresiva donde dice nuevo chat». El arreglo vive en el frontend; aquí, el contrato entre repos (qué lee la regla, la
cuenta regresiva en su sitio y sus textos en los cuatro idiomas) y el verbo del prompt del aviso.
"""
from __future__ import annotations

import json
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


# ── El verbo del aviso ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("comida,verbo,infinitivo", [
    ("Desayuno", "desayunaste", "desayunar"), ("Almuerzo", "almorzaste", "almorzar"),
    ("Merienda", "merendaste", "merendar"), ("Cena", "cenaste", "cenar"),
])
def test_el_aviso_pide_el_verbo_de_su_comida(comida, verbo, infinitivo):
    import proactive_agent as pa
    from prompts.proactive import PROACTIVE_PROMPT
    assert pa.INFINITIVO_DE_COMIDA[comida] == infinitivo
    prompt = PROACTIVE_PROMPT.format(
        missing_meal=comida, verbo=pa.VERBO_DE_COMIDA[comida], infinitivo=infinitivo, trigger_time="5:30 PM",
        diet_type="balanceada", goals="ganar músculo", tone_instruction="", style_instruction="",
    )
    # [P1-PLAN-LOTE-150] El aviso llega ANTES de la comida, así que el prompt manda ANIMAR. Lo que este test
    # protege sigue igual: que el verbo sea el de SU comida y nunca el de otra.
    assert f"«{infinitivo}»" in prompt
    assert f"«¿Ya {verbo}?»" in prompt
    assert "nunca el de otra" in prompt


def test_el_bucle_pasa_el_verbo_al_prompt():
    import proactive_agent as pa
    assert "verbo=VERBO_DE_COMIDA.get(meal_to_check," in _src("proactive_agent.py")
    assert set(pa.VERBO_DE_COMIDA) == {"Desayuno", "Almuerzo", "Merienda", "Cena"}


# ── La renovación diaria (frontend) ───────────────────────────────────────────────────────────────

def test_la_regla_mira_el_dia_anotado_y_el_ultimo_mensaje():
    regla = _front("src/utils/chatSessionDay.js")
    cuerpo = regla.split("export const debeRenovarse", 1)[1].split("\n};", 1)[0]
    assert "const anotado = diaAnotadoDe(sessionId);" in cuerpo
    assert "if (!anotado || anotado >= hoy) return false;" in cuerpo
    assert "ultimoMensajeReal(messages)" in cuerpo
    assert "minutosDeGracia * 60000" in cuerpo
    assert "export const MINUTOS_DE_GRACIA = 15;" in regla


def test_el_agente_renueva_al_volver_y_con_el_reloj_sin_pisar_nada():
    agente = _front("src/pages/AgentPage.jsx")
    assert "P1-PLAN-LOTE-73" in agente
    bloque = agente.split("const renovarChatDelDia = useStableCallback(", 1)[1].split("}, [renovarChatDelDia]);", 1)[0]
    for pieza in (
        "isTurnActiveRef.current",
        "draftSnapshotRef.current",
        "debeRenovarse({ messages: messagesRef.current, sessionId: currentSessionIdRef.current })",
        "abrirSesionAutomatica()",
        "_setCurrentSessionId(nuevoId);",
        "adopcionDelDiaHechaRef.current = false;",
        "fetchChatSessions();",
        "document.addEventListener('visibilitychange', alVolver);",
        "setInterval(() => renovarChatDelDia('reloj'), 60 * 1000)",
    ):
        assert pieza in bloque, pieza
    # Hidratar no renueva el día.
    assert "marcarActividad(currentSessionId, diaDeActividad(messages, currentSessionId));" in agente


def test_la_cuenta_regresiva_esta_bajo_el_boton_y_la_cabecera_conserva_su_alto():
    barra = _front("src/components/agent/SidebarRecientes.jsx")
    assert "import CuentaRegresivaChat from './CuentaRegresivaChat';" in barra
    assert "</button>\n                <CuentaRegresivaChat />\n            </div>" in barra
    assert "padding: '0.75rem 1rem 0.5rem'" in barra
    cuenta = _front("src/components/agent/CuentaRegresivaChat.jsx")
    assert "height: '1rem'" in cuenta and "lineHeight: '1rem'" in cuenta
    assert "0.75rem + 2.75rem + 0.25rem + 1rem + 0.5rem" in _front("src/pages/AgentPage.jsx")


def test_los_textos_estan_en_los_cuatro_idiomas():
    claves = (
        "Nuevo chat automático en {h} h {m} min",
        "Nuevo chat automático en {m} min",
        "Nuevo chat automático en menos de un minuto",
        "El chat se renueva solo cada día a medianoche, si no estás escribiendo.",
    )
    for loc in ("en-US", "pt-BR", "fr-FR", "it-IT"):
        cat = json.loads(_front(f"src/i18n/locales/{loc}.json"))
        for k in claves:
            assert cat.get(k), (loc, k)
            if "{h}" in k:
                assert "{h}" in cat[k] and "{m}" in cat[k], (loc, k)


def test_las_pruebas_de_la_pagina_existen():
    prueba = _front("src/__tests__/Agent.renovacion_diaria.test.jsx")
    assert "al volver a la pestaña pasada la medianoche, se abre el chat de hoy" in prueba
    assert "abrir hoy un chat viejo desde Recientes no lo renueva al volver a la pestaña" in prueba


def test_marcador_y_documento():
    assert "P1-PLAN-LOTE-73" in _src("app.py")
    doc = _src("docs/chat_sesion_del_dia.md")
    assert "P1-PLAN-LOTE-73" in doc and "diaDeActividad" in doc
    assert "P1-PLAN-LOTE-73" in _src("docs/recordatorios_de_comida.md")
