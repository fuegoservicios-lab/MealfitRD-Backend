"""[P1-PLAN-LOTE-58 · 2026-09-15] El anuncio de la acción antes de la tool, repetido después.

Batería del coach v6 (docs/coach_bateria_2026_09_15.md): el modelo escribe «Te las añado a tu Nevera.»,
llama a la tool y confirma «Listo ✅ Ya están en tu Nevera…». La frase de anuncio se quita del texto
previo a la PRIMERA tool en el stream y en el texto final (el `done` y lo que se graba), a la vez.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

_BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_BACKEND))

import agent as A  # noqa: E402


def test_marcador_del_lote():
    app = (_BACKEND / "app.py").read_text(encoding="utf-8")
    m = re.search(r'_LAST_KNOWN_PFIX = "P1-PLAN-LOTE-(\d+) · (\d{4}-\d{2}-\d{2})"', app)
    assert m and int(m.group(1)) >= 58 and m.group(2) >= "2026-09-15"
    assert "[P1-PLAN-LOTE-58 · 2026-09-15]" in app


# Los tres casos reales de la batería v6 (texto previo a la tool) y lo que debe quedar.
@pytest.mark.parametrize("antes,queda", [
    ("¡Buenas esas zanahorias, Angelo! Te las añado a tu Nevera.", "¡Buenas esas zanahorias, Angelo!"),
    ("Anoto la alergia en tu perfil de inmediato.", ""),
    ("¡Bien ahí, Angelo! 🙌 Te lo anoto tal como venía en el plan:", "¡Bien ahí, Angelo!"),
    ("¡Bién, Angelo! Zanahoria y cilantro fresquito — perfecto para mañana 🥕\n\n"
     "Los sumo a tu Nevera para que no se te olviden:",
     "¡Bién, Angelo! Zanahoria y cilantro fresquito — perfecto para mañana 🥕"),
    ("Lo anoto...", ""),
    ("Tienes toda la razón, disculpa — te lo guardo ahora mismo para que no vuelva a salir pescado.",
     "Tienes toda la razón, disculpa"),
    ("Voy a registrarlo ahora.", ""),
    ("Déjame guardarlo en tu perfil.", ""),
])
def test_se_quita_el_anuncio_y_queda_lo_demas(antes, queda):
    assert A._strip_tool_announcement(antes) == queda


@pytest.mark.parametrize("texto", [
    "Asumo pan de molde y 40 g de queso por sándwich.",   # de dónde sale el estimado
    "Te anoto los 2 vasos.",                               # con cifra: contenido, no se toca
    "¿Eso fue tu cena de hoy?",
    "Registro de hoy: vas bien con la proteína.",          # «Registro» sustantivo en medio... se queda si no es la última
    "Ojo: tu plan de hoy trae maní en el desayuno.",
    "Listo, lo agrego.",                                    # empieza por «Listo»: fuera del patrón a propósito
])
def test_el_contenido_no_se_toca(texto):
    assert A._strip_tool_announcement(texto) == texto.rstrip()


def _turno(pre: str, post: str, con_tool: bool = True) -> list:
    tc = [{"name": "modify_pantry_inventory", "args": {}, "id": "c1"}] if con_tool else []
    return [HumanMessage(content="mira lo que compré"),
            AIMessage(content=pre, tool_calls=tc),
            ToolMessage(content="¡Despensa actualizada!", tool_call_id="c1"),
            AIMessage(content=post)]


def test_el_texto_final_pierde_el_anuncio_como_el_stream():
    out = A._build_final_content_from_messages(
        _turno("¡Buenas esas zanahorias! Te las añado a tu Nevera.", "Listo ✅ Ya están en tu Nevera."))
    assert out == "¡Buenas esas zanahorias!\n\nListo ✅ Ya están en tu Nevera."


def test_un_anuncio_puro_desaparece_entero():
    out = A._build_final_content_from_messages(
        _turno("Anoto la alergia en tu perfil de inmediato.", "Listo, quedó guardada tu alergia a la leche."))
    assert out == "Listo, quedó guardada tu alergia a la leche."


def test_sin_tool_no_se_toca_nada():
    """Solo el texto de la AIMessage que LLEVA la tool_call: es lo único que el stream retiene."""
    msgs = [HumanMessage(content="hola"), AIMessage(content="Lo anoto y te digo cómo va tu día.")]
    assert A._build_final_content_from_messages(msgs) == "Lo anoto y te digo cómo va tu día."


def test_la_respuesta_final_nunca_pierde_su_anuncio():
    """La ÚLTIMA pasada (sin tool) es la respuesta: aunque termine en «lo anoto», se queda."""
    out = A._build_final_content_from_messages(
        _turno("", "Quedó en tu Nevera. Si compras más, lo anoto.", con_tool=True))
    assert out.endswith("lo anoto.")


def test_el_knob_lo_apaga(monkeypatch):
    monkeypatch.setattr(A, "STRIP_TOOL_ANNOUNCE_ENABLED", False)
    assert A._strip_tool_announcement("Te las añado a tu Nevera.") == "Te las añado a tu Nevera."
    out = A._build_final_content_from_messages(_turno("Te las añado a tu Nevera.", "Listo."))
    assert "Te las añado" in out


def _turno_escritura(tool: str, texto: str) -> list:
    return [HumanMessage(content="x"),
            AIMessage(content="", tool_calls=[{"name": tool, "args": {}, "id": "t1"}]),
            ToolMessage(content="¡Éxito!", tool_call_id="t1"),
            AIMessage(content=texto)]


def test_el_nombre_de_un_boton_no_dispara_el_nudge_del_diario():
    """Mini-batería F7: tras guardar la alergia, «'Actualizar platos'» se leía como comida y el
    nudge reescribía la respuesta — el usuario la veía DOS veces en el stream."""
    texto = ("Anotado, Angelo: alergia a la **leche** guardada en tu perfil. Para actualizar los "
             "cambios, usa el botón **'Actualizar platos'** en la página Plan — ya el sistema tiene "
             "tu alergia registrada para futuros planes.")
    assert A.route_tools({"messages": _turno_escritura("update_form_field", texto)}) != "nudge_diary_tool"


def test_una_comida_de_verdad_sigue_disparando_aunque_haya_un_boton():
    texto = "Listo, tu cena quedó registrada con 520 kcal; si quieres, usa 'Cambiar Plato'."
    assert A.route_tools({"messages": _turno_escritura("log_water_glass", texto)}) == "nudge_diary_tool"


def test_el_stream_quita_el_anuncio_en_el_mismo_punto_que_emite_la_narracion():
    src = (_BACKEND / "agent.py").read_text(encoding="utf-8")
    i = src.index("# Narración corta: se emite (P1-CHAT-NARRATION-KEPT)")
    tramo = src[i:i + 700]
    assert "_retenido = _strip_tool_announcement(_retenido)" in tramo
    assert "if _retenido.strip():" in tramo
