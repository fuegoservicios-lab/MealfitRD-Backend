"""[P1-MODIFY-HONEST-FAIL · 2026-07-12] El fallo del chat-modify dice el motivo REAL.

Vivo (owner, con 64 items en la Nevera): "actualiza el desayuno por algo
variado" → ambos intentos (strict + expansión) fallaron el VALIDADOR DE
MACROS (carbs 70g vs 37g target, slot de ~380 kcal) — pero la tool devolvía
un mensaje hardcodeado "FALLO POR INVENTARIO INSUFICIENTE ... carece de los
ingredientes adecuados", y el agente le contó al usuario que sus ingredientes
estaban "al borde de agotarse / reservados". Falso: la Nevera estaba llena y
cero reservas activas (verificado en DB). Un fallo mal atribuido = usuario
desconfiando de su Nevera.

Fix: mensaje reason-aware — si el error fue de macros, lo dice (con el conteo
real de la despensa y sugerencias direccionales) y PROHÍBE la narrativa de
escasez; el mensaje de inventario queda solo para fallos de inventario.
tooltip-anchor: P1-MODIFY-HONEST-FAIL
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "tools.py"), encoding="utf-8") as f:
    _TL = f.read()


def _fail_block():
    i = _TL.find("P1-MODIFY-HONEST-FAIL")
    assert i != -1, "el fallo reason-aware desapareció"
    return _TL[i:i + 3000]


def test_macro_failure_is_not_blamed_on_pantry():
    blk = _fail_block()
    assert '"MACROS FUERA" in _err_txt' in blk, \
        "el clasificador debe reconocer el error del validador de macros"
    assert "FALLO POR CONVERGENCIA DE MACROS" in blk
    assert "NO por falta de ingredientes" in blk
    assert "PROHIBIDO decirle que le faltan ingredientes" in blk, \
        "sin la prohibición explícita, el LLM re-inventa la narrativa de escasez"


def test_pantry_count_travels_for_honesty():
    blk = _fail_block()
    assert "_n_pantry = len(clean_ingredients)" in blk, \
        "el conteo real de la despensa viaja en el mensaje (64 items ≠ escasez)"


def test_inventory_message_still_exists_for_real_scarcity():
    blk = _fail_block()
    assert "FALLO POR INVENTARIO INSUFICIENTE" in blk, \
        "el mensaje de inventario se conserva para fallos genuinos de despensa"
