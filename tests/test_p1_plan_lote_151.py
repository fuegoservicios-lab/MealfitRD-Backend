# -*- coding: utf-8 -*-
"""[P1-PLAN-LOTE-151 · 2026-09-21] El aviso que llega ANTES no puede abrir preguntando si ya comiste.

El lote 150 movió el recordatorio a 15 min ANTES de la hora habitual y le dijo al prompt que animara. Pero le dejó
una salida escrita: «si aun así necesitas preguntar, el verbo sigue siendo el suyo: "¿Ya {verbo}?"». Medido en
producción el mismo día (21-sep, 11:30 local, usuario del dueño): el mensaje que salió empezaba por
«¿Ya desayunaste? Si aún no, este es el momento ideal…». El modelo tomó la excepción como la norma.

*Una excepción escrita en un prompt no es una excepción: es una opción, y el modelo la elige.* Si de verdad no
quieres una conducta, el prompt tiene que prohibirla, no permitirla «solo si hace falta».

El verbo propio de cada comida (lote 73) se conserva: sigue en el prompt, ahora dentro de la prohibición, así que
el ancla de aquel test sigue hablando del mismo contrato.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]


def _src(nombre: str) -> str:
    return (_BACKEND / nombre).read_text(encoding="utf-8")


def test_el_prompt_prohibe_preguntar_si_ya_comio():
    prompt = _src("prompts/proactive.py")
    assert "PROHIBIDO abrir preguntando «¿Ya {verbo}?»" in prompt
    # la salida que el modelo usaba como norma ya no está
    assert "Si aun así necesitas preguntar" not in prompt
    assert "si aún así necesitas preguntar" not in prompt.lower()


def test_sigue_mandando_animar_con_el_verbo_de_esa_comida():
    """Lo del lote 73 no se pierde al cerrar la salida: el verbo sigue siendo el de SU comida."""
    prompt = _src("prompts/proactive.py")
    assert "«{infinitivo}»" in prompt
    assert "nunca el de otra" in prompt
    assert "anímale a comer ahora" in prompt
    # y no vuelve a dar por hecho que ya comió
    assert "No des por hecho que ya comió" not in prompt or "PROHIBIDO" in prompt


@pytest.mark.parametrize("comida,infinitivo", [
    ("Desayuno", "desayunar"), ("Almuerzo", "almorzar"), ("Merienda", "merendar"), ("Cena", "cenar"),
])
def test_el_prompt_se_arma_sin_huecos(comida, infinitivo):
    """`.format()` con todas las claves: un placeholder nuevo sin su argumento reventaría el cron a las 6 a.m."""
    import proactive_agent as pa
    from prompts.proactive import PROACTIVE_PROMPT

    assert pa.INFINITIVO_DE_COMIDA[comida] == infinitivo
    # Se rellenan TODOS los huecos que declare la plantilla, no una lista a mano: así, si mañana nace otro
    # placeholder, este test sigue midiendo lo suyo en vez de romperse por un `KeyError` ajeno.
    import string

    claves = {n for _l, n, _s, _c in string.Formatter().parse(PROACTIVE_PROMPT) if n}
    valores = {k: "" for k in claves}
    valores.update(
        missing_meal=comida,
        verbo=pa.VERBO_DE_COMIDA[comida],
        infinitivo=infinitivo,
        trigger_time="8:45 AM",
    )
    texto = PROACTIVE_PROMPT.format(**valores)
    assert "{" not in texto.replace("{{", "").replace("}}", ""), "quedó un placeholder sin rellenar"
    assert infinitivo in texto


def test_el_marcador_va_con_su_lote():
    app = _src("app.py")
    m = re.search(r'_LAST_KNOWN_PFIX\s*=\s*"P1-PLAN-LOTE-(\d+) · \d{4}-\d{2}-\d{2}"', app)
    assert m, "el marcador cambió de forma"
    assert int(m.group(1)) >= 151, "el marcador nunca baja"
