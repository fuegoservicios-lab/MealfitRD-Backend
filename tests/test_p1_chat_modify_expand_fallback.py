"""[P1-CHAT-MODIFY-EXPAND-FALLBACK · 2026-07-12] El chat no se rinde donde el botón encuentra la vuelta.

Vivo (owner): "actualiza el desayuno" + "algo variado" → modify_single_meal
corrió PANTRY-STRICT (default), no convergió con solo lo de la Nevera y el
chat se rindió ("no cuajó sin salirse de lo que tienes físicamente") — el
flujo del botón no muere así.

Fix en la intercepción de execute_tools (agent.py): si el intento strict no
produjo `modified_meal`, reintenta UNA vez con allow_pantry_expansion=True
(equivale a aceptar comprar 1-2 ingredientes extra) con transparencia doble:
toast vía coherence_warnings + instrucción en el ToolMessage para que el coach
lo diga con naturalidad. El retry NO corre si el caller ya pidió expansión.
tooltip-anchor: P1-CHAT-MODIFY-EXPAND-FALLBACK
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "agent.py"), encoding="utf-8") as f:
    _AG = f.read()


def _block():
    # OJO: anclar en CÓDIGO único — el marker aparece antes en un comment del
    # clamp de timeouts (lección primera-ocurrencia, 4ª vez hoy).
    i = _AG.find("_expand_fallback_used = False")
    assert i != -1, "el fallback desapareció de execute_tools"
    return _AG[max(0, i - 2500):i + 4200]


def test_fallback_retries_expanded_once():
    blk = _block()
    assert "if not _allow_exp:" in blk, \
        "el retry SOLO cuando el caller no pidió ya expansión (sin loops)"
    assert blk.count("allow_pantry_expansion=True") == 1, "UN retry expandido"
    assert '"modified_meal" in _probe' in blk, \
        "fallo = sin modified_meal en el resultado (mismo sentinel del branch de éxito)"


def test_transparency_travels_both_channels():
    blk = _block()
    assert "coherence_warnings.append(" in blk, "toast al usuario (canal SSE done)"
    assert "fuera de tu Nevera" in blk
    # El aviso al coach viaja en el friendly string del éxito.
    assert "_expand_fallback_used" in _AG
    assert "se suman a su lista de compras" in _AG


def test_prompts_teach_no_surrender():
    with open(os.path.join(_BACKEND, "prompts", "chat_agent.py"), encoding="utf-8") as f:
        prompts = f.read()
    assert prompts.count("allow_pantry_expansion=true") >= 2, \
        "ambos builders enseñan cuándo pedir expansión explícita"
    assert "auto-reintenta" in prompts or "AUTOMÁTICAMENTE" in prompts
