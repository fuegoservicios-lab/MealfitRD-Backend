"""[P1-DIARY-CLAIM-VERIFY · 2026-07-31] El coach no puede decir "registrada" en falso.

## El incidente

Turno real (`corr=610fd9c8`). El usuario escribió "cene dos panes con queso" y
el coach respondió:

    Cena registrada. Asumo 2 panes de molde (~60g c/u) con 40g de queso...
    Quedó anotada como tu cena de hoy.

El journal de ese turno completo son DOS líneas: `call_model` y `Finalizado con
éxito`. Ni un `execute_tools`, ni un `🍽️ [DIARY]`, ni una fila en
`consumed_meals`. El panel "Progreso en Tiempo Real" seguía en 0 comidas.

El modelo NARRÓ el registro. `build_tools_instructions_stream` ya se lo prohíbe
literalmente ("NUNCA digas 'lo registro' o 'anotado' si no llamaste la
herramienta en ese turno") — pero eso es una instrucción, no un control. El tier
`free` enruta a `deepseek-v4-flash` (todo lo que no sea basic/plus/ultra cae al
barato, fail-cheap por diseño) y flash se salta la llamada con bastante más
frecuencia que pro.

Lo que se pierde no es solo la comida: el diario queda mintiendo hacia atrás. El
coach de mañana lee "no registró nada" y le reprocha al usuario algo que sí hizo.

## Qué protege este test

1. Que el guard DISPARE cuando hay afirmación sin llamada.
2. Que NO dispare cuando el coach es honesto ("no pude registrarlo") — un guard
   que castiga la honestidad empuja al modelo justo al comportamiento que
   queremos evitar.
3. Que NO dispare cuando la tool sí se llamó.
4. Que el reintento tenga TOPE, y que su flag esté declarado en `ChatState`:
   LangGraph descarta las claves que un nodo devuelve y no existen en el schema,
   así que un flag no declarado se pierde en silencio y el turno entra en bucle
   `call_model → nudge → call_model` quemando tokens hasta el timeout.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from agent import (  # noqa: E402
    ChatState,
    _diary_tool_called_this_turn,
    _reply_claims_diary_write,
    nudge_diary_tool,
    route_tools,
)

# El texto EXACTO del incidente. Si un refactor del regex deja de reconocerlo,
# el bug que motivó este fix vuelve a pasar silencioso.
RESPUESTA_DEL_INCIDENTE = (
    "Cena registrada. Asumo 2 panes de molde (~60g c/u) con 40g de queso en total.\n"
    "~400 kcal\nP: 18g · C: 48g · G: 15g\n"
    "Quedó anotada como tu cena de hoy. Si el estimado no cuadra con lo que "
    "comiste, ajústalo o bórralo desde 'Progreso en Tiempo Real'."
)


# --------------------------------------------------- detección de la afirmación

@pytest.mark.parametrize("texto", [
    RESPUESTA_DEL_INCIDENTE,
    "Listo, quedó registrado.",
    "Ya te lo anoté en el diario.",
    "Te lo apunté como merienda.",
    "Cena registrada.",
])
def test_detecta_la_afirmacion_de_registro(texto):
    assert _reply_claims_diary_write(texto) is True


@pytest.mark.parametrize("texto", [
    "No pude registrarlo, hazlo desde 'Progreso en Tiempo Real'.",
    "No quedó anotada porque falta el tipo de comida.",
    "Todavía no te lo he registrado: ¿fue almuerzo o cena?",
    "Sin registrar aún — dime cuántos panes fueron.",
    "Nunca te registré esa comida, revisa el panel.",
])
def test_no_dispara_cuando_el_coach_es_HONESTO(texto):
    """Un guard que castiga la honestidad empuja al modelo a mentir."""
    assert _reply_claims_diary_write(texto) is False


@pytest.mark.parametrize("texto", [
    "",
    "Buen provecho. ¿Cómo te sentiste después?",
    "Te recomiendo pescado al horno con ensalada.",
])
def test_no_dispara_en_respuestas_sin_afirmacion(texto):
    assert _reply_claims_diary_write(texto) is False


# ------------------------------------------- ¿corrió la tool en ESTE turno?

def _ai_con_tool(nombre: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[
        {"name": nombre, "args": {}, "id": "call_1"}
    ])


def test_reconoce_la_tool_llamada_en_el_turno():
    msgs = [
        HumanMessage(content="cene dos panes con queso"),
        _ai_con_tool("log_consumed_meal"),
        AIMessage(content="Cena registrada."),
    ]
    assert _diary_tool_called_this_turn(msgs) is True


def test_una_llamada_del_turno_ANTERIOR_no_cuenta():
    """El corte es el último HumanMessage: si no, una comida de ayer avalaría
    la afirmación de hoy y el guard quedaría inerte tras el primer registro."""
    msgs = [
        HumanMessage(content="me comí un sandwich"),
        _ai_con_tool("log_consumed_meal"),
        AIMessage(content="Registrado."),
        HumanMessage(content="cene dos panes con queso"),
        AIMessage(content="Cena registrada."),
    ]
    assert _diary_tool_called_this_turn(msgs) is False


def test_otra_tool_cualquiera_no_avala_el_registro():
    msgs = [
        HumanMessage(content="cene dos panes con queso"),
        _ai_con_tool("check_current_pantry"),
        AIMessage(content="Cena registrada."),
    ]
    assert _diary_tool_called_this_turn(msgs) is False


# ------------------------------------------------------------- el enrutado

def _estado(messages, **extra) -> dict:
    return {"messages": messages, "user_id": "u-1", **extra}


def test_el_turno_del_incidente_se_desvia_al_reintento():
    estado = _estado([
        HumanMessage(content="cene dos panes con queso"),
        AIMessage(content=RESPUESTA_DEL_INCIDENTE),
    ])
    assert route_tools(estado) == "nudge_diary_tool"


def test_si_la_tool_corrio_el_turno_termina_normal():
    estado = _estado([
        HumanMessage(content="cene dos panes con queso"),
        _ai_con_tool("log_consumed_meal"),
        AIMessage(content=RESPUESTA_DEL_INCIDENTE),
    ])
    assert route_tools(estado) == "__end__"


def test_una_respuesta_con_tool_calls_sigue_yendo_a_execute_tools():
    """El camino normal no se toca."""
    estado = _estado([
        HumanMessage(content="cene dos panes con queso"),
        _ai_con_tool("log_consumed_meal"),
    ])
    assert route_tools(estado) == "execute_tools"


def test_el_reintento_ocurre_UNA_sola_vez():
    """Sin tope, el turno cicla call_model→nudge→call_model hasta el timeout."""
    estado = _estado([
        HumanMessage(content="cene dos panes con queso"),
        AIMessage(content=RESPUESTA_DEL_INCIDENTE),
    ], diary_claim_retried=True)
    assert route_tools(estado) == "__end__"


def test_el_nudge_marca_el_flag_y_reinyecta_la_exigencia():
    salida = nudge_diary_tool(_estado([
        HumanMessage(content="cene dos panes con queso"),
        AIMessage(content=RESPUESTA_DEL_INCIDENTE),
    ]))
    assert salida["diary_claim_retried"] is True
    (msg,) = salida["messages"]
    assert isinstance(msg, SystemMessage)
    assert "log_consumed_meal" in msg.content
    # Y deja salida honesta: si de verdad no había nada que registrar, el modelo
    # no debe inventarse una comida para satisfacer al guard.
    assert re.search(r"NO hay nada que registrar", msg.content)


# --------------------------------------------------- el flag existe de verdad

def test_el_flag_esta_declarado_en_ChatState():
    """LangGraph DESCARTA las claves que un nodo devuelve y no están en el
    schema. Sin esta declaración el tope no existe: `route_tools` leería
    siempre False y el turno entraría en bucle."""
    assert "diary_claim_retried" in ChatState.__annotations__


def test_el_grafo_conoce_el_nodo_del_reintento():
    """Sanity del vehículo: los tests de arriba llaman a `route_tools` directo;
    si el nodo no está cableado en el grafo, todos pasarían y en producción
    LangGraph reventaría por destino desconocido."""
    from agent import chat_builder
    assert "nudge_diary_tool" in chat_builder.nodes
