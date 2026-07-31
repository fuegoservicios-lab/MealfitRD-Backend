"""[P1-CHAT-DELIBERATION-HIDDEN · 2026-07-31] La deliberación no se muestra.

## El incidente

El usuario escribió "cene dos panes con queso" y ANTES de la respuesta le
aparecieron ~4.000 caracteres de deliberación en primera persona: "Hmm, pero son
las 10:23 AM", "Espera, según la regla 6-bis", "Déjame pensar"… y entre ellos la
frase que lo delata:

    «Déjame llamar la herramienta primero (regla de cero texto antes de
    herramienta)»

El modelo CITA la regla que se lo prohíbe mientras la incumple. Una regla en el
prompt no es un control.

## Lo que NO era (descartado midiendo)

- No es un leak de reasoning tokens: DeepSeek los manda en `reasoning_content`,
  un campo aparte que el loop del stream no lee.
- El thinking además está desactivado desde `P1-DEEPSEEK-THINKING-OFF`
  (verificado en el cliente de producción: `extra_body={'thinking':
  {'type': 'disabled'}}`).
- Tampoco es que falte el guard: `if not msg_chunk.tool_calls` está escrito. Lo
  derrota el ORDEN del streaming — los chunks de texto llegan antes que la
  tool_call, así que al evaluarse todavía no hay `tool_calls`.

## Por qué el corte es por LONGITUD

Descartar todo el texto pre-tool desharía `P1-CHAT-NARRATION-KEPT`
(2026-07-28), que restauró a propósito la narración corta ("Lo anoto…") porque
antes aparecía en vivo y luego se desvanecía — pérdida de dato real. Dos guardas
sobre el mismo campo oscilan.

Una narración legítima son ~30-60 chars; la deliberación del incidente pasaba de
4.000. El umbral (300) vive en medio de un hueco de dos órdenes de magnitud: no
es un número peleado.

## Las dos mitades

Retener en el stream NO basta: `_build_final_content_from_messages` alimenta el
evento `done` y el `save_message` del router, así que sin filtrar también ahí el
texto volvería un segundo más tarde — y quedaría GRABADO en el historial.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from langchain_core.messages import AIMessage, HumanMessage  # noqa: E402

import agent as A  # noqa: E402

DELIBERACION = (
    "El usuario dice que cenó dos panes con dos lonjas de queso gouda. Hmm, pero son "
    "las 10:23 AM. Esto es extraño — ¿cenó a esta hora? Quizás su horario rotativo lo "
    "hace comer a horas raras. Pero según la regla 6-bis, debo derivar el slot de la "
    "hora cruzada con qué comidas del día faltan. Espera — déjame pensar. La regla 8 "
    "dice que el día por defecto es HOY. Déjame llamar la herramienta primero (regla "
    "de cero texto antes de herramienta). Voy a estimar los macros y registrar."
)
NARRACION_CORTA = "Lo anoto y te digo cómo va tu día."
RESPUESTA = "Listo, te lo dejé anotado como cena: 2 panes con queso (~300 kcal)."


# ----------------------------------------------------- la frontera longitud

def test_la_deliberacion_del_incidente_supera_el_cap():
    cap = A._chat_pretool_narration_max_chars()
    assert len(DELIBERACION) > cap, (
        f"la deliberación real ({len(DELIBERACION)} chars) no supera el cap ({cap}) "
        f"— el guard no la cazaría"
    )


def test_la_narracion_legitima_queda_MUY_por_debajo_del_cap():
    """Si el umbral estuviera peleado, mover el knob rompería un caso u otro."""
    cap = A._chat_pretool_narration_max_chars()
    assert len(NARRACION_CORTA) * 4 < cap, (
        f"la narración legítima ({len(NARRACION_CORTA)} chars) está demasiado cerca "
        f"del cap ({cap}): el umbral debería vivir en un hueco amplio, no en un borde"
    )


def test_el_knob_existe_y_viene_encendido():
    assert A._chat_hold_pretool_text() is True


# ------------------------------- segunda mitad: el `done` y lo que se GRABA

def _turno(*partes: str) -> list:
    msgs: list = [HumanMessage(content="cene dos panes con dos lonjas de queso gooda")]
    msgs += [AIMessage(content=p) for p in partes]
    return msgs


def test_la_deliberacion_NO_llega_al_texto_final_del_turno():
    """Es lo que alimenta el evento `done` y el `save_message` del router."""
    final = A._build_final_content_from_messages(_turno(DELIBERACION, RESPUESTA))
    assert RESPUESTA in final
    assert "Hmm" not in final and "regla 6-bis" not in final, (
        "P1-CHAT-DELIBERATION-HIDDEN: la deliberación sobrevive en el texto final "
        "del turno. Retenerla solo en el stream no sirve: reaparecería en el `done` "
        "y quedaría grabada en el historial."
    )


def test_la_narracion_CORTA_sigue_sobreviviendo(monkeypatch):
    """Anti-oscilación: P1-CHAT-NARRATION-KEPT no se deshace.

    Este es el test que impide que 'arreglar' la deliberación se lleve por
    delante el fix de hace tres días.
    """
    final = A._build_final_content_from_messages(_turno(NARRACION_CORTA, RESPUESTA))
    assert NARRACION_CORTA in final, (
        "se perdió la narración corta — eso es exactamente el bug que "
        "P1-CHAT-NARRATION-KEPT cerró el 2026-07-28"
    )
    assert RESPUESTA in final


def test_un_turno_de_UN_solo_bloque_largo_NO_se_vacia():
    """El último bloque nunca se descarta: una respuesta larga y legítima
    (sin herramienta de por medio) tiene que llegar entera."""
    largo = "Te explico con calma. " * 40
    final = A._build_final_content_from_messages(_turno(largo))
    assert final.strip(), "un turno de un solo bloque quedó VACÍO"
    assert "Te explico" in final


def test_apagar_el_knob_devuelve_la_deliberacion(monkeypatch):
    """El kill switch tiene que funcionar de verdad: si solo apagara el
    stream y no esta mitad, revertir dejaría el sistema en un tercer estado
    que nadie probó."""
    monkeypatch.setenv("MEALFIT_CHAT_HOLD_PRETOOL_TEXT", "false")
    final = A._build_final_content_from_messages(_turno(DELIBERACION, RESPUESTA))
    assert "regla 6-bis" in final


# --------------------------------------- primera mitad: el loop del stream

@pytest.mark.parametrize("fragmento", [
    "_hold_pretool = _chat_hold_pretool_text()",
    "if _hold_pretool and not _tool_seen:",
    "_pretool_buf.append(chunk_content)",
])
def test_el_loop_del_stream_retiene_antes_de_emitir(fragmento):
    """Parser-based sobre el generador: no se puede instanciar el grafo aquí,
    pero sí exigir que el buffer exista y se use ANTES del `yield`."""
    src = (BACKEND / "agent.py").read_text(encoding="utf-8")
    assert fragmento in src, f"falta en el loop del stream: {fragmento!r}"


def test_el_stream_vacia_el_buffer_si_no_hubo_herramienta():
    """Sin este flush, un turno conversacional normal saldría VACÍO — el guard
    se comería justo lo que debe proteger. Es el modo de fallo más caro de
    todo este cambio, así que va anclado."""
    src = (BACKEND / "agent.py").read_text(encoding="utf-8")
    assert "if _pretool_buf and not _tool_seen:" in src, (
        "no hay flush final del buffer: un turno sin tool_call no emitiría nada"
    )
