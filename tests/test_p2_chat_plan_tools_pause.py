"""[P2-CHAT-PLAN-TOOLS-PAUSE · 2026-08-15] Los dos caminos del coach que seguían
mandando al usuario en pausa a una pantalla y a un plan que su modo no tiene.

H15 · EL BULLET DE LAS TOOLS APAGADAS
    Con `MEALFIT_CHAT_PLAN_TOOLS_ENABLED` en OFF (el default), el prompt le dice
    al modelo: «dile que use los botones de la página Plan — 'Cambiar Plato' en
    cada comida, o 'Actualizar platos'». En modo contador esa página no está en
    la nav (se rotula «Hoy») y esos botones no existen: se manda al usuario a
    buscar controles inexistentes, y de paso se le insinúa que su plan sigue
    gobernando el día.

    El arreglo sigue el patrón que ya usa `build_inventory_context`: el caller
    —que es quien sabe el modo— pasa el veredicto. Este módulo es de PROMPTS y no
    debe tocar la DB.

H16 · EL NUDGE DE LAS 23:00
    Era el único camino que no pasa por `_plan_context_for_chat`. Su mensaje al
    usuario que no registró nada ofrece: «¿restamos lo de hoy de tu nevera como si
    lo hubieras cocinado?». «Lo de hoy» presupone un plan que prescribió algo — en
    modo contador no hay tal cosa, y restar de la Nevera «como si lo hubieras
    cocinado» no tiene referente. El nudge en sí SÍ es apropiado en contador
    (registrar comidas es exactamente lo que ese modo hace); lo que sobra es la
    oferta anclada al plan.

Tooltip-anchor: P2-CHAT-PLAN-TOOLS-PAUSE
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACK = Path(__file__).resolve().parent.parent
_PROACTIVE = _BACK / "proactive_agent.py"


@pytest.fixture()
def prompts():
    from prompts import chat_agent
    return chat_agent


# ---------------------------------------------------------------------------
# H15 · el bullet de tools
# ---------------------------------------------------------------------------

def test_en_pausa_el_bullet_no_manda_a_la_pagina_plan(prompts, monkeypatch):
    monkeypatch.setattr(prompts, "_plan_tools_enabled", lambda: False, raising=True)
    for fn in (prompts._plan_tools_bullets_inline, prompts._plan_tools_bullets_stream):
        texto = fn(plan_en_pausa=True)
        assert "página Plan" not in texto, (
            f"[P2-CHAT-PLAN-TOOLS-PAUSE] `{fn.__name__}` sigue mandando a la «página "
            "Plan» con el plan en pausa. En modo contador esa entrada de nav se "
            "rotula «Hoy» y los botones de plato no existen."
        )
        assert "Cambiar Plato" not in texto and "Actualizar platos" not in texto


def test_en_pausa_el_bullet_ofrece_la_puerta_que_si_existe(prompts, monkeypatch):
    monkeypatch.setattr(prompts, "_plan_tools_enabled", lambda: False, raising=True)
    texto = prompts._plan_tools_bullets_inline(plan_en_pausa=True)
    assert re.search(r"reanud", texto, re.IGNORECASE), (
        "[P2-CHAT-PLAN-TOOLS-PAUSE] El bullet de pausa no menciona reanudar. "
        "Quitar la redirección mala sin dar la buena deja al usuario sin salida: "
        "el Historial es su puerta de vuelta (P1-TRACKING-WINS)."
    )


def test_con_el_plan_activo_el_bullet_es_el_de_siempre(prompts, monkeypatch):
    monkeypatch.setattr(prompts, "_plan_tools_enabled", lambda: False, raising=True)
    texto = prompts._plan_tools_bullets_inline(plan_en_pausa=False)
    assert "página Plan" in texto, (
        "[P2-CHAT-PLAN-TOOLS-PAUSE] Se perdió la redirección correcta del modo "
        "plan. Con el plan vigente esos botones SÍ existen y son la respuesta."
    )


def test_las_dos_variantes_conservan_la_prohibicion(prompts, monkeypatch):
    """Lo que NO puede caerse: que el agente no toca el plan de ninguna manera."""
    monkeypatch.setattr(prompts, "_plan_tools_enabled", lambda: False, raising=True)
    for pausa in (True, False):
        texto = prompts._plan_tools_bullets_inline(plan_en_pausa=pausa)
        assert "NO PUEDES modificar el plan" in texto
        assert "NUNCA prometas modificar el plan" in texto


def test_el_veredicto_viaja_desde_el_caller():
    """agent.py pasa `plan_en_pausa` en los CUATRO call sites (2 por path)."""
    src = (_BACK / "agent.py").read_text(encoding="utf-8")
    codigo = re.sub(r"^\s*#.*$", "", src, flags=re.MULTILINE)
    for fn in ("build_tools_instructions", "build_tools_instructions_stream"):
        pelados = re.findall(rf"{fn}\(user_id\)(?!\s*,)", codigo)
        assert not pelados, (
            f"[P2-CHAT-PLAN-TOOLS-PAUSE] Quedan llamadas a `{fn}(user_id)` sin "
            "`plan_en_pausa`. Este módulo es de PROMPTS y no lee la DB: si el caller "
            "no pasa el veredicto, el bullet vuelve a mandar a la página Plan."
        )


# ---------------------------------------------------------------------------
# H16 · el nudge de las 23:00
# ---------------------------------------------------------------------------

def test_el_nudge_nocturno_conoce_el_modo():
    fuente = _PROACTIVE.read_text(encoding="utf-8")
    assert "plan_mode" in fuente or "_plan_mode_for_chat" in fuente, (
        "[P2-CHAT-PLAN-TOOLS-PAUSE] `proactive_agent.py` no mira el modo en ninguna "
        "línea. Era el ÚNICO camino del coach que no pasa por "
        "`_plan_context_for_chat`, y su nudge de «no registraste nada» ofrece "
        "«restamos lo de hoy de tu nevera como si lo hubieras cocinado» — «lo de "
        "hoy» presupone un plan que en modo contador no existe."
    )


def test_la_oferta_de_restar_de_la_nevera_esta_gateada():
    # Sin comentarios: el comentario que EXPLICA el gate cita la frase, y buscarla
    # en crudo encontraba la prosa antes que el código. Un guard que mide su propia
    # documentación no mide nada.
    fuente = re.sub(r"^\s*#.*$", "", _PROACTIVE.read_text(encoding="utf-8"), flags=re.MULTILINE)
    i_oferta = fuente.find("como si lo hubieras cocinado")
    assert i_oferta > 0, (
        "[P2-CHAT-PLAN-TOOLS-PAUSE] Desapareció la oferta de restar de la Nevera; "
        "si fue intencional, actualiza este guard."
    )
    ventana = fuente[max(0, i_oferta - 2500):i_oferta]
    assert re.search(r"plan_mode|en_pausa|tracking", ventana), (
        "[P2-CHAT-PLAN-TOOLS-PAUSE] La oferta de restar de la Nevera «como si lo "
        "hubieras cocinado» no está gateada por modo. En contador no hay plan del "
        "que restar: la frase no tiene referente."
    )


def test_el_nudge_sigue_existiendo_en_modo_contador():
    """Registrar comidas ES lo que hace el contador: el nudge no se apaga."""
    fuente = _PROACTIVE.read_text(encoding="utf-8")
    assert "no ha registrado NINGUNA comida" in fuente, (
        "[P2-CHAT-PLAN-TOOLS-PAUSE] Se eliminó el nudge nocturno entero. Lo que "
        "sobraba era la oferta anclada al plan, no el recordatorio: registrar "
        "comidas es exactamente para lo que sirve el modo contador."
    )
