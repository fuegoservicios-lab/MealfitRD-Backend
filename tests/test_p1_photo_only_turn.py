# -*- coding: utf-8 -*-
"""[P1-PHOTO-ONLY-TURN · 2026-09-06] Una foto sin texto dejaba el turno del usuario VACÍO.

Caso vivo del 06-sep a las 16:48. El dueño abre un chat nuevo y sube la foto de un almuerzo
dominicano, sin escribir nada. **El escáner acertó de lleno** — «arroz blanco, espaguetis
guisados con salsa de tomate y aceitunas, carne de res guisada y plátano maduro frito…
(1065 kcal, 51 g de proteína)», persistido en `attachments[0].description` — y
`build_vision_context` compuso su bloque y llegó al system prompt como debe.

Y aun así el coach contestó con el menú del día y terminó preguntando «¿Almorzaste algo distinto
al ceviche?». No miró la foto.

**El análisis existía y nadie lo leía**, por un turno vacío. Un mensaje de usuario sin una sola
palabra no es una pregunta: el modelo llena el hueco con lo que mejor encaja al abrir sesión, que
es saludar y recitar el plan. La instrucción del bloque de foto («actúa proactivamente y resume
lo detectado») estaba ahí y perdió contra ese vacío.

Dos decisiones del arreglo, y este test existe sobre todo por ellas:

  · **El marcador es un emoji.** `P3-I18N-PROMPT-VISION-CLIENTE-ESPANOL` sacó del turno del
    usuario los cuatro bloques en español precisamente porque eran la señal más fuerte hacia el
    español. Escribir ahora «El usuario subió una foto» sería deshacer aquello. Un emoji no tiene
    idioma.
  · **No se persiste.** Lo que se guarda en `agent_messages` sigue siendo el texto del usuario
    (vacío), así que la burbuja del chat no cambia. Lo que el usuario VE y lo que el modelo LEE
    no tienen por qué ser lo mismo.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import pytest  # noqa: E402

_SRC = (_BACKEND / "routers" / "chat.py").read_text(encoding="utf-8")


def _bloque() -> str:
    i = _SRC.find("_prompt_para_el_modelo = prompt")
    assert i > 0, "desapareció el marcador del turno de solo-foto"
    return _SRC[i:i + 700]


def test_el_turno_de_solo_foto_no_llega_vacio():
    b = _bloque()
    assert 'if not str(prompt or "").strip()' in b, "dejó de detectar el turno sin texto"
    assert 'vision.get("kind")' in b, "el marcador ya no exige que haya foto"


def test_el_marcador_no_tiene_idioma():
    """Media hora de este P-fix se fue en NO deshacer P3-I18N-PROMPT-VISION-CLIENTE-ESPANOL. Si
    alguien cambia el emoji por una frase, el modelo vuelve a tener una señal fuerte hacia el
    español dentro del turno del usuario — que es justo lo que aquel arreglo quitó."""
    b = _bloque()
    m = re.search(r'_prompt_para_el_modelo = "([^"]*)"', b)
    assert m, "no se encontró el valor del marcador"
    valor = m.group(1)
    assert valor in ("\\U0001f4f7", "\U0001f4f7"), f"el marcador dejó de ser el emoji: {valor!r}"
    assert not re.search(r"[a-zA-Z]{4,}", valor), "el marcador tiene palabras: eso es idioma"


def test_el_marcador_no_se_persiste():
    """`save_message_with_attachments` recibe `prompt`, no el marcador: la burbuja del chat sigue
    mostrando lo que el usuario escribió (nada)."""
    i = _SRC.find("save_message_with_attachments(")
    assert i > 0
    llamada = _SRC[i:i + 260]
    assert "prompt" in llamada and "_prompt_para_el_modelo" not in llamada, (
        "el marcador se está persistiendo: aparecerá en la burbuja del usuario")


def test_el_agente_recibe_el_marcador():
    """La otra mitad: si el marcador se calcula y no se pasa, el arreglo es letra muerta — que es
    exactamente lo que le pasó al análisis de la foto."""
    i = _SRC.find("chat_with_agent_stream(")
    assert i > 0
    llamada = _SRC[i:i + 420]
    assert "prompt=_prompt_para_el_modelo" in llamada, (
        "el agente sigue recibiendo el prompt vacío")


def test_un_turno_con_texto_no_se_toca():
    """Con texto, el turno es del usuario y punto. Meterle un emoji delante sería ruido."""
    b = _bloque()
    assert "_prompt_para_el_modelo = prompt" in b, "se perdió el valor por defecto"


def test_sin_foto_no_hay_marcador():
    """Un mensaje vacío SIN foto no debe recibir el emoji: no habría nada que mirar y el modelo
    respondería a una imagen que no existe."""
    b = _bloque()
    i_cond = b.find("if not str(prompt")
    i_and = b.find("vision", i_cond)
    assert 0 < i_cond < i_and, "la condición dejó de exigir la foto además del texto vacío"


# ── el bloque de contexto sigue siendo lo que el modelo tiene que obedecer ────────────────────
def test_el_contexto_de_foto_pide_actuar_cuando_no_hay_texto():
    """El emoji da al modelo un turno al que responder; el bloque le dice QUÉ hacer. Sin esta
    instrucción, el turno no vacío tampoco garantizaría que hable de la foto."""
    from prompts.chat_agent import build_vision_context
    ctx = build_vision_context({
        "kind": "multi",
        "has_text": False,
        "items": [{"kind": "plato", "description": "arroz blanco, carne guisada y plátano"}],
    })
    assert "PLATO/COMIDA" in ctx and "arroz blanco" in ctx
    assert "proactiv" in ctx.lower(), "se perdió la orden de actuar sin que el usuario escriba"


def test_el_contexto_con_texto_no_fuerza_el_resumen():
    """Con texto del usuario, la orden de «resume lo detectado» estorbaría: el usuario ya dijo
    qué quería saber."""
    from prompts.chat_agent import build_vision_context
    ctx = build_vision_context({
        "kind": "multi",
        "has_text": True,
        "items": [{"kind": "plato", "description": "arroz blanco"}],
    })
    assert "proactiv" not in ctx.lower()
