"""[P1-CHAT-NARRATION-KEPT · 2026-07-28] La narración streameada de un pase
narrate-then-act (content + tool_calls en la MISMA completion) se descartaba.

Bug original (verificado en `agent.py` antes de este fix):
  Cuando el modelo emite AIMessage con `content` (narración: "Lo anoto...")
  Y `tool_calls` en el mismo turno, la narración YA se streamea al cliente
  como eventos `chunk` ordinarios (`chat_with_agent_stream`, el loop
  `for event in stream_iter`). El grafo ejecuta la tool y vuelve a
  `call_model`, que produce una SEGUNDA AIMessage ("Listo, quedó anotado").
  Al final, el evento `done` (y el path non-stream `chat_with_agent`)
  tomaban SOLO `final_messages[-1]` — la narración YA leída por el usuario
  desaparecía del payload persistido. `routers/chat.py::save_message`
  persiste justo ese `response`, así que un reload del historial NUNCA
  mostraba la primera narración: pérdida de dato real, no solo visual.

Fix:
  `_build_final_content_from_messages` (agent.py) reconstruye el texto
  final uniendo TODAS las AIMessage con contenido no vacío emitidas DESPUÉS
  del último HumanMessage, en orden, con de-duplicación verbatim entre
  pasadas. Usado por AMBOS paths (`chat_with_agent_stream` Y el path
  non-stream `chat_with_agent` — el historial de este repo tiene un patrón
  documentado de fixes que aterrizan en un path espejo y se olvidan del
  otro, así que este test cubre el helper compartido directamente en vez
  de re-implementar la lógica en el test).

  Además: reglas de brevedad/densidad (P1-CHAT-BREVITY) añadidas a los 4
  prompts base del chat vía la constante compartida `_CHAT_BREVITY_RULES`
  (prompts/chat_agent.py) — "decir cada punto una sola vez", "una sola
  pregunta por respuesta", "no anunciar una acción antes de ejecutarla".
  Caso real (owner, foto de "Los Tres Golpes" a las 10pm): la MISMA
  advertencia repetida 3 veces en 4 párrafos + un párrafo "El problema:"
  que solo reformulaba lo mismo + DOS preguntas compitiendo al cierre.

tooltip-anchor: P1-CHAT-NARRATION-KEPT
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import _build_final_content_from_messages

_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_APP_PY = _BACKEND_ROOT / "app.py"
_AGENT_PY = _BACKEND_ROOT / "agent.py"
_CHAT_AGENT_PROMPTS_PY = _BACKEND_ROOT / "prompts" / "chat_agent.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ===========================================================================
# 1. narrate-then-act: AMBOS textos de AIMessage sobreviven, en orden
# ===========================================================================

def test_narrate_then_act_keeps_both_ai_texts_in_order():
    messages = [
        HumanMessage(content="me desayuné mangú con salami"),
        AIMessage(
            content="Lo anoto...",
            tool_calls=[{"name": "log_consumed_meal", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(content="ok registrado", tool_call_id="call_1"),
        AIMessage(content="Listo, quedó anotado"),
    ]

    result = _build_final_content_from_messages(messages)

    assert "Lo anoto..." in result
    assert "Listo, quedó anotado" in result
    # Orden: la narración PRIMERO, la confirmación DESPUÉS — cualquier otro
    # orden significa que el join no respetó la secuencia real del turno.
    assert result.index("Lo anoto...") < result.index("Listo, quedó anotado")


# ===========================================================================
# 2. Turno de una sola AIMessage (Q&A plano) — sin regresión
# ===========================================================================

def test_single_ai_message_turn_unchanged():
    messages = [
        HumanMessage(content="¿cuántas calorías tiene el mangú?"),
        AIMessage(content="El mangú tiene aproximadamente 220 kcal por taza."),
    ]

    result = _build_final_content_from_messages(messages)

    assert result == "El mangú tiene aproximadamente 220 kcal por taza."


# ===========================================================================
# 3. De-duplicación verbatim entre pasadas
# ===========================================================================

def test_verbatim_repetition_across_passes_is_deduplicated():
    messages = [
        HumanMessage(content="agrega 2 libras de pollo a mi nevera"),
        AIMessage(
            content="Listo, lo agrego.",
            tool_calls=[{"name": "modify_pantry_inventory", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(content="ok", tool_call_id="call_1"),
        # El modelo repite EXACTAMENTE la misma frase en la segunda pasada
        # (verbatim, incluido espaciado tras strip) — no debe duplicarse.
        AIMessage(content="Listo, lo agrego."),
    ]

    result = _build_final_content_from_messages(messages)

    assert result == "Listo, lo agrego."
    assert result.count("Listo, lo agrego.") == 1


def test_verbatim_repetition_with_surrounding_whitespace_is_deduplicated():
    """La de-dup compara tras strip() — un modelo que reemite la misma
    frase con espacios/saltos de línea extra al inicio/fin sigue contando
    como repetición verbatim (no como narración nueva)."""
    messages = [
        HumanMessage(content="tomé un vaso de agua"),
        AIMessage(
            content="Anotado.",
            tool_calls=[{"name": "log_water_glass", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(content="ok", tool_call_id="call_1"),
        AIMessage(content="  Anotado.  \n"),
    ]

    result = _build_final_content_from_messages(messages)

    assert result.count("Anotado.") == 1


# ===========================================================================
# 4. Mensajes ANTES del último HumanMessage no contaminan el turno actual
# ===========================================================================

def test_messages_before_last_human_message_are_excluded():
    messages = [
        # Turno anterior — su AIMessage NO debe aparecer en el turno actual.
        HumanMessage(content="hola, quiero bajar de peso"),
        AIMessage(content="Perfecto, cuéntame más sobre tu rutina."),
        # Turno actual.
        HumanMessage(content="me comí un mangú"),
        AIMessage(content="Registrado: mangú con salami."),
    ]

    result = _build_final_content_from_messages(messages)

    assert "Registrado: mangú con salami." in result
    assert "Perfecto, cuéntame más sobre tu rutina." not in result


# ===========================================================================
# 5. Casos borde estructurales
# ===========================================================================

def test_empty_messages_list_returns_empty_string():
    assert _build_final_content_from_messages([]) == ""


def test_ai_messages_with_empty_content_are_skipped():
    """AIMessage(content='') que solo trae tool_calls (patrón normal del
    loop narrate-then-act cuando el modelo NO narra, solo actúa) no debe
    producir una entrada vacía / espacio extra en el resultado."""
    messages = [
        HumanMessage(content="agrega leche a mi nevera"),
        AIMessage(
            content="",
            tool_calls=[{"name": "modify_pantry_inventory", "args": {}, "id": "call_1"}],
        ),
        ToolMessage(content="ok", tool_call_id="call_1"),
        AIMessage(content="Listo, agregada la leche."),
    ]

    result = _build_final_content_from_messages(messages)

    assert result == "Listo, agregada la leche."


def test_list_content_blocks_are_flattened_to_text():
    """Algunos providers devuelven `content` como lista de bloques
    estructurados ([{'type': 'text', 'text': '...'}]) en vez de un string
    plano — mismo tratamiento que el código pre-fix aplicaba inline."""
    messages = [
        HumanMessage(content="hola"),
        AIMessage(content=[{"type": "text", "text": "Hola, ¿en qué te ayudo?"}]),
    ]

    result = _build_final_content_from_messages(messages)

    assert result == "Hola, ¿en qué te ayudo?"


def test_no_human_message_uses_entire_list_as_tail():
    """Si por algún motivo no hay NINGÚN HumanMessage en la lista (no
    debería pasar en producción, pero el helper debe degradar con
    gracia), trata la lista entera como el tramo del turno actual."""
    messages = [
        AIMessage(content="Primero"),
        AIMessage(content="Segundo"),
    ]

    result = _build_final_content_from_messages(messages)

    assert "Primero" in result
    assert "Segundo" in result


# ===========================================================================
# 6. AMBOS code paths (stream y non-stream) usan el helper compartido —
#    cierra el modo de fallo histórico "fix en un path espejo, olvidado en
#    el otro" documentado en CLAUDE.md.
# ===========================================================================

def test_both_call_sites_use_the_shared_helper(agent_src: str = None):
    src = _read(_AGENT_PY)
    # Debe haber exactamente 2 callsites productivos del helper (uno por
    # path) además de su propia definición — si un futuro refactor vuelve
    # a inlinear `final_messages[-1]` en cualquiera de los dos, este count
    # cae a 1 y el test debe fallar loud.
    call_count = len(re.findall(r"_build_final_content_from_messages\(", src))
    assert call_count >= 3, (
        "esperaba la definición + al menos 2 callsites (stream y "
        "non-stream) de _build_final_content_from_messages en agent.py, "
        f"encontré {call_count} — ¿un path volvió a usar messages[-1] "
        "directo?"
    )
    # Anti-regresión explícita: el patrón viejo `final_messages[-1]` no
    # debe sobrevivir en ninguno de los dos call sites que este fix tocó.
    assert "last_msg = final_messages[-1]" not in src, (
        "regresión: el path non-stream volvió a tomar solo el último "
        "mensaje — pierde la narración de un pase narrate-then-act."
    )


# ===========================================================================
# 7. Prompts: las 4 constantes cargan las reglas de brevedad — falla si
#    se olvida alguna (el failure mode explícito del brief).
# ===========================================================================

@pytest.fixture(scope="module")
def chat_prompts():
    from prompts.chat_agent import (
        CHAT_SYSTEM_PROMPT_BASE,
        CHAT_STREAM_SYSTEM_PROMPT_BASE,
        CHAT_AGENT_INLINE_PROMPT,
        CHAT_STREAM_INLINE_PROMPT,
    )
    return {
        "CHAT_SYSTEM_PROMPT_BASE": CHAT_SYSTEM_PROMPT_BASE,
        "CHAT_STREAM_SYSTEM_PROMPT_BASE": CHAT_STREAM_SYSTEM_PROMPT_BASE,
        "CHAT_AGENT_INLINE_PROMPT": CHAT_AGENT_INLINE_PROMPT,
        "CHAT_STREAM_INLINE_PROMPT": CHAT_STREAM_INLINE_PROMPT,
    }


_BREVITY_MARKERS = (
    "REGLAS DE BREVEDAD Y DENSIDAD",
    "DI CADA PUNTO UNA SOLA VEZ",
    "UNA SOLA PREGUNTA POR RESPUESTA",
    # [P1-CHAT-FILLER-STRIP · 2026-07-30] Era "NO ANUNCIES UNA ACCIÓN ANTES DE EJECUTARLA".
    # Renombrada porque estaba escrita como LISTA NEGRA DE FRASES ("lo registro", "lo anoto") y el
    # modelo la esquivaba con sinónimos: en una respuesta real la incumplió 3 veces sin usar ninguna
    # de las frases prohibidas ("Vamos a registrarlo", "Dame un segundo…", "Registrando…"). Ahora es
    # una regla de COMPORTAMIENTO: cero texto delante de la llamada, sea como sea que lo formules.
    # Este test hizo su trabajo — pilló el renombre en los 4 prompts a la vez, que es exactamente
    # para lo que existe (que una edición no pueda arreglar 3 de 4).
    "CERO TEXTO ANTES DE UNA HERRAMIENTA",
    # La regla del slot es nueva (P1-DIARY-DINNER-SLOT): un `snack` a las 22:47 con la cena sin
    # registrar era la cena. Se ancla igual para que no se caiga de uno de los 4 prompts.
    "EL SLOT SE DERIVA DE LOS DATOS",
)


def test_all_four_prompts_carry_every_brevity_marker(chat_prompts: dict):
    missing = {}
    for name, prompt_text in chat_prompts.items():
        absent = [m for m in _BREVITY_MARKERS if m not in prompt_text]
        if absent:
            missing[name] = absent
    assert not missing, (
        f"prompt(s) sin las reglas de brevedad completas: {missing} — "
        f"las 4 constantes (CHAT_SYSTEM_PROMPT_BASE, "
        f"CHAT_STREAM_SYSTEM_PROMPT_BASE, CHAT_AGENT_INLINE_PROMPT, "
        f"CHAT_STREAM_INLINE_PROMPT) deben cargar el bloque completo."
    )


def test_brevity_rules_preserve_future_intent_asking_distinction(chat_prompts: dict):
    """La regla de brevedad NO debe resucitar preguntas para frases en
    PASADO (P1-CHAT-ACT-DONT-ASK sigue vivo) — debe declarar explícitamente
    que preguntar sigue siendo correcto para FUTURO/intención."""
    for name, prompt_text in chat_prompts.items():
        assert "PASADO" in prompt_text and "FUTURO" in prompt_text, (
            f"[{name}] la regla de brevedad no distingue pasado (actuar sin "
            f"preguntar) de futuro/intención (preguntar sigue siendo "
            f"correcto) — riesgo de resucitar preguntas para acciones ya "
            f"confirmadas por el usuario."
        )


def test_brevity_rules_never_say_to_cut_numbers(chat_prompts: dict):
    """La brevedad recorta repetición, NUNCA los números clínicos.

    Ancla DENTRO del bloque de brevedad específicamente (desde el marker
    'REGLAS DE BREVEDAD Y DENSIDAD' en adelante) — NO en el prompt entero.
    Los 3 prompts con sección "REGLAS DE FORMATO VISUAL" ya mencionan
    "kcal"/"proteína" en su ejemplo de negritas (**350 kcal**), así que
    buscar en el prompt completo daría un falso-positivo (pasaría aunque
    la cláusula 5 del bloque de brevedad se borrara por completo en esos
    3 prompts) y solo detectaría la regresión en CHAT_SYSTEM_PROMPT_BASE
    (el único de los 4 sin esa sección)."""
    marker = "REGLAS DE BREVEDAD Y DENSIDAD"
    for name, prompt_text in chat_prompts.items():
        idx = prompt_text.find(marker)
        assert idx != -1, f"[{name}] no encontré el marker de brevedad"
        brevity_block = prompt_text[idx:]
        assert "kcal" in brevity_block and (
            "proteína" in brevity_block or "proteina" in brevity_block
        ), (
            f"[{name}] falta, DENTRO del bloque de brevedad, la cláusula "
            f"que preserva kcal/macros — la brevedad no debe poder "
            f"interpretarse como licencia para cortar datos clínicos."
        )


def test_brevity_rules_defined_once_shared_constant():
    """Las 4 constantes deben concatenar la MISMA constante compartida
    (`_CHAT_BREVITY_RULES`), no 4 copias independientes del texto — así
    una futura edición no puede arreglar 3 de 4 y dejar el cuarto stale."""
    src = _read(_CHAT_AGENT_PROMPTS_PY)
    assert src.count("_CHAT_BREVITY_RULES") >= 5, (
        "esperaba 1 definición + 4 usos (uno por prompt) de "
        "_CHAT_BREVITY_RULES en prompts/chat_agent.py — si el count bajó, "
        "alguna constante dejó de compartir el bloque."
    )


# ===========================================================================
# 8. Marker `_LAST_KNOWN_PFIX` — supersession-proof (mismo patrón que
#    P1-CHAT-ACT-DONT-ASK y demás P-fixes recientes del repo)
# ===========================================================================

_MARKER_RE = re.compile(r'_LAST_KNOWN_PFIX\s*=\s*"([^"]+)"')


def test_marker_bumped_or_superseded():
    src = _read(_APP_PY)
    m = _MARKER_RE.search(src)
    assert m, "no encontré _LAST_KNOWN_PFIX en app.py"
    marker = m.group(1)
    if "P1-CHAT-NARRATION-KEPT" in marker:
        return
    fecha = re.search(r"(\d{4}-\d{2}-\d{2})", marker)
    assert fecha and fecha.group(1) >= "2026-07-28", (
        f"marker {marker!r} es ANTERIOR a P1-CHAT-NARRATION-KEPT "
        f"(2026-07-28): sin bump un operador no puede confirmar que este "
        f"fix está vivo en producción."
    )


def test_marker_slug_has_this_test_file():
    src = _read(_APP_PY)
    m = _MARKER_RE.search(src)
    assert m
    marker = m.group(1)
    if "P1-CHAT-NARRATION-KEPT" not in marker:
        pytest.skip("marker actual pertenece a un P-fix posterior")
    slug = marker.split("·", 1)[0].strip().replace("-", "_").lower()
    assert slug == "p1_chat_narration_kept"
    assert Path(__file__).name.startswith(f"test_{slug}")
