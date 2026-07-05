"""[P1-CRITIQUE-VARIETY-CONTRACT · 2026-07-05] El corrector del self-critique aprende variedad.

Medido en vivo (corrida b1b95537, 3 intentos completos quemados): el self-critique detectó
"queso blanco repetido en 3 días" y su corrector lo "arregló" metiendo REVOLTILLOS en los 3
días → SOBREUSO DE HUEVO (5/12), 'revoltillo' ×3 días y misma-proteína-same-day → rechazo tras
rechazo. Los autofixes de huevo declinaron correctamente (protegen platos PROTAGONISTAS —
revoltillo/tortilla son identidad de huevo): el fallo era aguas arriba. Causa: el prompt del
corrector solo veía SU día y CERO reglas de variedad — whack-a-mole garantizado.

Fix (prompt-only, cero costo extra): el `correction_prompt` inyecta (a) el CONTRATO DE VARIEDAD
exacto de los gates (cap de huevo 3 + jamás same-day, misma proteína same-day, plato-base 3+
días, fruta same-day) y (b) un resumen de los PLATOS DE LOS OTROS DÍAS para que la corrección
no colisione cross-day (el corrector corrigiendo el Día 2 no puede saber que los Días 1 y 3 ya
son revoltillos sin verlos).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def _prompt_block():
    i = _GO.index("Eres un nutricionista chef. Corrige SOLO el Día")
    return _GO[i:i + 3500]


def test_marker_anchored():
    assert "P1-CRITIQUE-VARIETY-CONTRACT" in _GO


def test_variety_contract_in_correction_prompt():
    blk = _prompt_block()
    assert "CONTRATO DE VARIEDAD" in blk
    assert "máximo 3 comidas con huevo" in blk, "cap global de huevo (el gate rechaza 4+)"
    assert "NO resuelvas una repetición metiendo huevo/revoltillo" in blk, \
        "la instrucción anti-whack-a-mole exacta del incidente b1b95537"
    assert "MISMO plato-base" in blk or "MISMO plato-base".lower() in blk.lower()
    assert "2+ comidas del MISMO día" in blk


def test_other_days_summary_injected():
    i = _GO.index("[P1-CRITIQUE-VARIETY-CONTRACT · 2026-07-05]")
    win = _GO[i:i + 2500]
    assert "_other_days_block" in win, "el corrector debe VER los platos de los otros días"
    assert "PLATOS DE LOS OTROS DÍAS" in win
    assert "{_other_days_block}" in _prompt_block(), "el bloque viaja dentro del prompt"


def test_contract_before_target_day_json():
    """El contrato va ANTES del JSON del día (el LLM lee las reglas antes del payload)."""
    blk = _prompt_block()
    assert blk.index("CONTRATO DE VARIEDAD") < blk.index("actual (JSON)")
