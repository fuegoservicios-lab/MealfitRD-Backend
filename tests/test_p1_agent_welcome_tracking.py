"""[P1-AGENT-WELCOME-TRACKING · 2026-08-14 · mitad backend] El contexto del chat
deja de jurarle al modelo que el plan está «activo» cuando el usuario lo pausó.

EL BUG, en dos mitades. La visible la reportó el dueño con captura: con la
generación de planes desactivada, el agente abría con «De cena para hoy tienes:
Pasta Integral Salteada…» — la comida del plan PAUSADO como si gobernara el día.
Esa mitad vive en el saludo del frontend (`AgentPage.jsx`) y tiene su propio test.

Esta es la mitad invisible y más profunda: los DOS paths del chat (stream y
no-stream) inyectaban el plan al system prompt con el literal «tiene este plan de
comidas activo» / «Plan activo». Con eso, aunque el saludo callara, la primera
pregunta del usuario («¿qué me toca hoy?») recibía el plan pausado como
prescripción vigente. El modelo hace lo que el contexto le jura.

POR QUÉ EL BLOQUE NO SE OMITE en modo contador, que sería lo cómodo: el usuario
puede preguntar por su plan pausado («¿qué tenía mi plan?», «¿lo reanudo?») y el
agente debe poder responder — y ofrecer reanudar, que es la puerta de vuelta que
P1-PLAN-MODE dejó abierta. Datos presentes, encuadre correcto: es la misma
decisión que tomó `_prune_plan_for_chat` (podar, no amputar).

POR QUÉ UN HELPER (`_plan_context_for_chat`) y no dos edits inline: los dos paths
ya divergieron una vez en zona horaria («la divergencia entre ambos ya ha causado
bugs», dice el propio P1-CHAT-PAST-DAYS), y un encuadre que vive en dos f-strings
divergirá igual. Además un helper puro se testea con monkeypatch; dos f-strings
dentro de funciones de 500 líneas, no.

Tooltip-anchor: P1-AGENT-WELCOME-TRACKING
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_AGENT = Path(__file__).resolve().parent.parent / "agent.py"


@pytest.fixture()
def agente():
    import agent
    return agent


def _con_modo(monkeypatch, agente, modo):
    """El lector de modo, controlado. El helper no debe tocar la DB en tests."""
    monkeypatch.setattr(
        agente, "_plan_mode_for_chat", lambda user_id: modo, raising=True,
    )


PLAN = {"days": [{"day": 1, "meals": [{"meal": "Cena", "name": "Pasta Integral Salteada"}]}]}


# ---------------------------------------------------------------------------
# 1. Comportamiento del helper
# ---------------------------------------------------------------------------

def test_con_plan_activo_el_encuadre_es_el_de_siempre(agente, monkeypatch):
    _con_modo(monkeypatch, agente, "plan")
    bloque = agente._plan_context_for_chat("u1", PLAN)
    assert "activo" in bloque.lower(), (
        "[P1-AGENT-WELCOME-TRACKING] Con el plan vigente, el contexto debe seguir "
        "presentándolo como activo — ese contrato no se toca."
    )
    assert "Pasta Integral Salteada" in bloque, "El plan viaja en el bloque."


def test_en_modo_contador_el_encuadre_dice_PAUSA(agente, monkeypatch):
    _con_modo(monkeypatch, agente, "tracking")
    bloque = agente._plan_context_for_chat("u1", PLAN)
    assert "pausa" in bloque.lower(), (
        "[P1-AGENT-WELCOME-TRACKING] En modo contador el contexto sigue jurando "
        "que el plan está activo. El modelo hace lo que el contexto le jura: con "
        "ese literal, «¿qué me toca hoy?» se contesta con un plan que el usuario "
        "PAUSÓ — exactamente el bug de la captura del dueño, un piso más abajo."
    )
    assert "activo" not in bloque.lower(), (
        "[P1-AGENT-WELCOME-TRACKING] El bloque de pausa no puede decir también "
        "«activo»: dos encuadres contradictorios dejan al modelo eligiendo uno."
    )


def test_en_pausa_el_plan_SIGUE_viajando(agente, monkeypatch):
    """Pausado ≠ amputado: «¿qué tenía mi plan?» debe poder responderse."""
    _con_modo(monkeypatch, agente, "tracking")
    bloque = agente._plan_context_for_chat("u1", PLAN)
    assert "Pasta Integral Salteada" in bloque, (
        "[P1-AGENT-WELCOME-TRACKING] El bloque de pausa amputó el plan. El usuario "
        "puede preguntar por él y el agente debe poder responder — y ofrecer "
        "reanudar, la puerta que P1-PLAN-MODE dejó abierta. Encuadre correcto con "
        "datos presentes, no silencio."
    )


def test_en_pausa_prohibe_presentarlo_como_lo_de_hoy(agente, monkeypatch):
    _con_modo(monkeypatch, agente, "tracking")
    bloque = agente._plan_context_for_chat("u1", PLAN)
    assert re.search(r"NO .*(present|recit|toca hoy|como si)", bloque, re.IGNORECASE), (
        "[P1-AGENT-WELCOME-TRACKING] Falta la instrucción negativa explícita. "
        "Decir «en pausa» sin prohibir el uso prescriptivo deja al modelo libre "
        "de seguir sirviéndolo de cena."
    )


def test_si_el_modo_no_se_puede_leer_cae_al_encuadre_activo(agente, monkeypatch):
    """Fail-open al comportamiento histórico: un fallo de DB no puede degradar
    el chat de TODOS los usuarios en modo plan (la inmensa mayoría)."""
    def revienta(user_id):
        raise RuntimeError("db caída")
    monkeypatch.setattr(agente, "_plan_mode_for_chat", revienta, raising=True)
    bloque = agente._plan_context_for_chat("u1", PLAN)
    assert "activo" in bloque.lower() and "Pasta Integral Salteada" in bloque


# ---------------------------------------------------------------------------
# 2. Los DOS paths consumen el helper (la divergencia ya costó bugs una vez)
# ---------------------------------------------------------------------------

def test_ambos_paths_usan_el_helper():
    src = _AGENT.read_text(encoding="utf-8")
    codigo = re.sub(r"#.*$", "", src, flags=re.MULTILINE)
    usos = codigo.count("_plan_context_for_chat(")
    # 1 definición + ≥2 llamadas (stream y no-stream).
    assert usos >= 3, (
        f"[P1-AGENT-WELCOME-TRACKING] `_plan_context_for_chat` tiene {usos} "
        "apariciones en el código; se esperaban la definición y una llamada por "
        "path (stream y no-stream). Si un path volvió a su f-string inline, los "
        "dos encuadres divergirán — igual que ya pasó con la zona horaria."
    )
    assert not re.search(r'\+= f"\\n\\nCONTEXTO CRÍTICO: El usuario actualmente tiene', codigo), (
        "[P1-AGENT-WELCOME-TRACKING] El path no-stream volvió al f-string inline "
        "con «plan de comidas activo» incondicional."
    )
