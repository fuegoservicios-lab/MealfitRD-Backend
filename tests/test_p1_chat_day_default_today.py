"""[P1-CHAT-DAY-DEFAULT-TODAY · 2026-07-30] Una comida contada en pasado SIN
día nombrado es de HOY — no es un caso indeterminado que haya que preguntar.

Incidente real (screenshots + SQL forense, 2026-07-30 ~13:37 RD):
    Usuario: "me comi dos panes con dos lonjas de queso gooda de desayuno".
    Coach:   "¿Eso fue hoy (Jueves) o fue de otro día? Si fue hoy, el
              desayuno de hoy ya lo tienes registrado de antes (3 panes con
              queso gouda, jamón, mayonesa y batida de mango)."
    Dashboard: "0 comidas registradas hoy".

Lo que la DB decía en ese instante (`consumed_meals`, user 55c537f9):
    - 2026-07-29T14:00:17Z  = 10:00 RD del MIÉRCOLES 29 → "3 Panes de Agua
      con Queso Gouda, Jamón de Pavo y Mayonesa + Batida de Mango" (1070 kcal)
    - 2026-07-30 local RD: CERO filas.

El prompt reconstruido con esos datos era **correcto** en los dos bloques:
`DIARIO DE HOY` decía "no ha registrado ninguna comida el día de hoy
todavía", y el desayuno aparecía fechado bajo "- Miércoles 29 jul:" en
`DIARIO REAL DE DÍAS ANTERIORES`. O sea: no fue un bug de datos ni de
ventana horaria. Fueron dos huecos de contrato en el prompt.

Hueco 1 — no existe un DÍA POR DEFECTO. La regla 7 de `_CHAT_BREVITY_RULES`
(P1-CHAT-DIARY-CORRECT, 2026-07-29) solo ofrece dos salidas: día explícito
o "única lectura razonable" → actuar; en cualquier otro caso → preguntar UNA
vez. Un mensaje en pasado sin fecha no encaja en la primera, así que cae en
"preguntar" — que es el caso MÁS común, no el excepcional. El propio test de
ayer ya avisó del riesgo ("sin esto la regla degenera en preguntar SIEMPRE")
y montó la guarda sobre el SLOT ("me comí el desayuno" ⇒ actuar); el DÍA se
quedó sin la suya. Nótese que aquí el usuario SÍ nombró el slot ("de
desayuno") y aun así le preguntaron: la guarda de ayer cubría la mitad del
par día/comida.

Hueco 2 — el pie de `build_past_diary_block` solo prohíbe la confusión
PLAN→diario ("nunca respondas con lo que el plan mandaba como si se lo
hubiera comido"). No dice nada sobre migrar una línea FECHADA de ese mismo
bloque al día de hoy, que es exactamente lo que pasó: el coach citó textual
la línea del Miércoles 29 y la presentó como "el desayuno de hoy ya lo
tienes registrado".

Este fix NO revierte P1-CHAT-DIARY-CORRECT: aquella regla prohíbe sacar el
día del TEMA de la pregunta anterior del coach, y sigue intacta. Esta acota
qué cuenta como "indeterminado" — sin default, ambas reglas oscilan sobre el
mismo campo (ayer: adivinó mal; hoy: pregunta de más).

Tooltip-anchor: P1-CHAT-DAY-DEFAULT-TODAY
"""
from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_PROMPTS_PY = _BACKEND_ROOT / "prompts" / "chat_agent.py"

# Marker de la regla nueva. Va en MAYÚSCULAS como el resto de las reglas del
# bloque, para que un grep por convención lo encuentre junto a las otras.
_DAY_DEFAULT_MARKER = "EL DÍA POR DEFECTO ES HOY"

# La regla de ayer que esta NO debe borrar (anti-oscilación).
_ATTRIBUTION_MARKER = "ATRIBUCIÓN DE DÍA/COMIDA"

_FOUR_PROMPT_NAMES = [
    "CHAT_SYSTEM_PROMPT_BASE",
    "CHAT_STREAM_SYSTEM_PROMPT_BASE",
    "CHAT_AGENT_INLINE_PROMPT",
    "CHAT_STREAM_INLINE_PROMPT",
]


@pytest.fixture(scope="module")
def prompts_src() -> str:
    return _PROMPTS_PY.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def four_base_prompts() -> dict:
    """Los 4 prompts BASE evaluados (no el texto crudo del archivo) — así un
    comentario de código nunca satisface el assert; solo cuenta lo que de
    verdad se le manda a la LLM."""
    from prompts.chat_agent import (
        CHAT_AGENT_INLINE_PROMPT,
        CHAT_STREAM_INLINE_PROMPT,
        CHAT_STREAM_SYSTEM_PROMPT_BASE,
        CHAT_SYSTEM_PROMPT_BASE,
    )
    return {
        "CHAT_SYSTEM_PROMPT_BASE": CHAT_SYSTEM_PROMPT_BASE,
        "CHAT_STREAM_SYSTEM_PROMPT_BASE": CHAT_STREAM_SYSTEM_PROMPT_BASE,
        "CHAT_AGENT_INLINE_PROMPT": CHAT_AGENT_INLINE_PROMPT,
        "CHAT_STREAM_INLINE_PROMPT": CHAT_STREAM_INLINE_PROMPT,
    }


def _rule_line(text: str, marker: str) -> str:
    """La LÍNEA que contiene el marker.

    Anclado a la ESTRUCTURA (cada regla numerada del bloque es una línea),
    no a una ventana de bytes fija: una ventana de N chars caduca sola en
    cuanto la regla de al lado crece un párrafo, y entonces el assert
    empieza a mirar texto que no es el suyo.
    """
    idx = text.find(marker)
    assert idx != -1, f"marker {marker!r} ausente"
    start = text.rfind("\n", 0, idx) + 1
    end = text.find("\n", idx)
    return text[start:] if end == -1 else text[start:end]


# ===========================================================================
# 1. La regla del día por defecto vive en LOS 4 prompts base.
# ===========================================================================

@pytest.mark.parametrize("prompt_name", _FOUR_PROMPT_NAMES)
def test_each_of_the_four_prompts_declares_today_as_default_day(
    four_base_prompts: dict, prompt_name: str
):
    """Parametrizado por constante: si alguien arregla 3 prompts y olvida el
    4to, falla el caso de ESE prompt y dice cuál."""
    text = four_base_prompts[prompt_name]
    assert _DAY_DEFAULT_MARKER in text, (
        f"[{prompt_name}] falta la regla del día por defecto "
        f"({_DAY_DEFAULT_MARKER!r}) — P1-CHAT-DAY-DEFAULT-TODAY. Sin ella, "
        f"un mensaje en pasado sin fecha cae en 'indeterminado' de la regla "
        f"de atribución y el coach pregunta '¿fue hoy o de otro día?' en el "
        f"caso más común que existe."
    )


@pytest.mark.parametrize("prompt_name", _FOUR_PROMPT_NAMES)
def test_rule_requires_a_positive_signal_to_open_the_day(
    four_base_prompts: dict, prompt_name: str
):
    """El día solo se considera abierto ante una señal POSITIVA de otro día.
    Sin esta mitad, la regla se lee como "asume hoy" a secas y se pierde el
    caso legítimo en que el usuario SÍ dijo 'ayer'."""
    rule = _rule_line(four_base_prompts[prompt_name], _DAY_DEFAULT_MARKER)
    assert "señal" in rule.lower() and "otro día" in rule, (
        f"[{prompt_name}] la regla no condiciona la pregunta a una señal "
        f"positiva de otro día. Regla recibida: {rule!r}"
    )
    assert "ayer" in rule.lower(), (
        f"[{prompt_name}] la regla no nombra el caso 'ayer' — el usuario lo "
        f"formuló así: 'Si le digo que es de ayer, es de ayer'."
    )


@pytest.mark.parametrize("prompt_name", _FOUR_PROMPT_NAMES)
def test_rule_forbids_quoting_a_previous_day_line_as_today(
    four_base_prompts: dict, prompt_name: str
):
    """Hueco 2 a nivel de regla general: una línea del diario de días
    anteriores NUNCA se describe como 'de hoy'."""
    rule = _rule_line(four_base_prompts[prompt_name], _DAY_DEFAULT_MARKER)
    assert "días anteriores" in rule.lower(), (
        f"[{prompt_name}] la regla no menciona el bloque de días anteriores, "
        f"que es de donde el coach sacó la línea que presentó como 'de hoy'. "
        f"Regla recibida: {rule!r}"
    )


# ===========================================================================
# 2. Arquitectura: 1 constante compartida, no 4 copias pegadas.
# ===========================================================================

def test_rule_lives_in_the_shared_constant_not_pasted_four_times(prompts_src: str):
    assert prompts_src.count(_DAY_DEFAULT_MARKER) == 1, (
        "P1-CHAT-DAY-DEFAULT-TODAY: el marker aparece más de una vez en el "
        "source — señal de que alguien pegó la regla por separado en vez de "
        "compartir `_CHAT_BREVITY_RULES`, reabriendo el riesgo de 'arreglar "
        "3 de 4'."
    )
    assert prompts_src.count("+ _CHAT_BREVITY_RULES") == 4


def test_marker_present_for_traceability(prompts_src: str):
    assert "P1-CHAT-DAY-DEFAULT-TODAY" in prompts_src


# ===========================================================================
# 3. Anti-oscilación: no se revierte P1-CHAT-DIARY-CORRECT ni
#    P1-CHAT-ACT-DONT-ASK. Las tres reglas coexisten y se acotan.
# ===========================================================================

@pytest.mark.parametrize("prompt_name", _FOUR_PROMPT_NAMES)
def test_previous_rules_survive_intact(four_base_prompts: dict, prompt_name: str):
    """Dos guardas sobre el mismo campo (el día del registro) escritas sin
    verse una a la otra OSCILAN: ayer el coach adivinaba el día desde el
    tema; hoy pregunta de más. El fix correcto ACOTA la de ayer, no la
    borra — por eso ambas deben seguir presentes."""
    text = four_base_prompts[prompt_name]
    assert _ATTRIBUTION_MARKER in text, (
        f"[{prompt_name}] P1-CHAT-DIARY-CORRECT fue borrada al añadir el "
        f"día por defecto — eso reabre el incidente de la fila fantasma "
        f"(el coach atribuyendo la respuesta al TEMA de su propia pregunta)."
    )
    assert "sigues actuando en el mismo turno sin pedir permiso" in text, (
        f"[{prompt_name}] P1-CHAT-ACT-DONT-ASK diluida."
    )
    # La salida "preguntar UNA vez" sigue existiendo para el caso realmente
    # ambiguo — la regla nueva la estrecha, no la elimina.
    assert "pregunta UNA vez" in text, (
        f"[{prompt_name}] desapareció la salida 'preguntar UNA vez' del caso "
        f"genuinamente ambiguo; sin ella el coach vuelve a adivinar en "
        f"silencio, que es el fallo de P1-CHAT-DIARY-CORRECT."
    )


def test_new_rule_does_not_mandate_asking_the_day(four_base_prompts: dict):
    """Negativo ACOTADO a la línea de la regla nueva (no al prompt entero:
    sobre texto largo un `not in` global choca con la prosa de otras reglas
    que sí hablan de preguntar, y el assert deja de significar nada)."""
    rule = _rule_line(
        four_base_prompts["CHAT_STREAM_SYSTEM_PROMPT_BASE"], _DAY_DEFAULT_MARKER
    )
    # Sanity del vehículo: si el slice viniera vacío, todos los negativos de
    # abajo pasarían en vacío y este test no protegería nada.
    assert len(rule) > 80 and _DAY_DEFAULT_MARKER in rule
    assert not re.search(r"pregunta\w*\s+(siempre|el día)", rule, re.IGNORECASE), (
        f"la regla del día por defecto no puede mandar preguntar el día — "
        f"es justo lo contrario de lo que arregla. Regla: {rule!r}"
    )


# ===========================================================================
# 4. Hueco 2, a nivel del bloque que se citó mal: el pie de
#    `build_past_diary_block` prohíbe migrar una línea fechada a "hoy".
# ===========================================================================

def _incident_rows() -> list:
    """Las filas REALES de la DB en el momento del incidente (UTC crudo,
    tal como las devuelve `get_consumed_meals_since`)."""
    return [
        {
            "meal_name": "3 Panes de Agua con Queso Gouda, Jamón de Pavo y Mayonesa + Batida de Mango",
            "calories": 1070, "meal_type": "desayuno",
            "consumed_at": "2026-07-29T14:00:17.402955+00:00",
        },
        {
            "meal_name": "Plátano Maduro Hervido con Huevos y Chuleta",
            "calories": 525, "meal_type": "almuerzo",
            "consumed_at": "2026-07-29T19:06:24.618648+00:00",
        },
        {
            "meal_name": "2 Sándwiches de Queso Gouda",
            "calories": 420, "meal_type": "cena",
            "consumed_at": "2026-07-30T02:47:20.611982+00:00",  # 22:47 RD del 29
        },
    ]


@pytest.fixture(scope="module")
def incident_block() -> str:
    from chat_history_context import build_past_diary_block
    return build_past_diary_block(
        _incident_rows(), date(2026, 7, 30), days_back=7, max_chars=3000,
        tz_offset_mins=240,
    )


def test_incident_block_dates_the_breakfast_to_wednesday(incident_block: str):
    """Sanity del vehículo + del bug: el bloque SÍ fechaba bien. Si esto
    fallara, el defecto sería de datos y el fix de prompt sería el
    equivocado."""
    assert "Miércoles 29 jul" in incident_block
    assert "3 Panes de Agua con Queso Gouda" in incident_block
    # La cena de las 22:47 RD del 29 pertenece al 29, no al 30.
    assert "Jueves 30 jul" not in incident_block, (
        "una comida de las 22:47 RD se está atribuyendo al día siguiente — "
        "regresión de P1-CHAT-PAST-DAYS-TZ-CONSUMED."
    )


def test_footer_forbids_presenting_a_previous_day_as_today(incident_block: str):
    """El pie del bloque debe cerrar el hueco día→día, no solo plan→diario."""
    assert "hoy" in incident_block.lower(), "el pie no menciona 'hoy' en absoluto"
    assert re.search(r"nunca.{0,120}como si fuera de hoy", incident_block,
                     re.IGNORECASE | re.DOTALL), (
        "P1-CHAT-DAY-DEFAULT-TODAY: el pie de `build_past_diary_block` no "
        "prohíbe presentar una comida de un día anterior como si fuera de "
        "hoy. Ese es el hueco por el que el coach citó textualmente la línea "
        "del Miércoles 29 y la llamó 'el desayuno de hoy'."
    )


def test_footer_keeps_the_plan_vs_real_guard(incident_block: str):
    """Anti-oscilación del pie: la guarda plan→diario de P1-CHAT-PAST-DAYS
    sigue viva junto a la nueva."""
    assert "El bloque del PLAN es lo prescrito" in incident_block
    assert "SIN REGISTRO" in incident_block


def test_footer_says_the_today_block_is_authoritative(incident_block: str):
    """La afirmación negativa del bloque `DIARIO DE HOY` ('no ha registrado
    ninguna comida hoy') es la verdad operativa. El coach la contradijo
    teniéndola en el mismo prompt; el pie debe declararla autoritativa."""
    assert "DIARIO DE HOY" in incident_block, (
        "el pie no remite al bloque `DIARIO DE HOY` como única fuente de lo "
        "registrado hoy — sin ese cruce explícito, los dos bloques se leen "
        "como dos listas sueltas y la de días anteriores gana por ser la "
        "más rica en detalle."
    )
