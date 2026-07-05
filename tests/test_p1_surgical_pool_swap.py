"""[P1-SURGICAL-POOL-SWAP · 2026-07-05] La regla de precedencia del corrector, por modo.

Caso vivo (corr=093ec919, primera corrida del retry quirúrgico): el 🩹 atribuyó los días
[2,3] y corrigió ambos en 14.8s, PERO la re-review rechazó con la MISMA razón (proteína
repetida). Causa raíz: la "REGLA DE PRECEDENCIA INVIOLABLE" original (P6-CRITIQUE-VS-SKELETON,
pensada para issues de slot) ordena MANTENER toda proteína asignada por el planificador — y el
pool del Día 2 incluía 'Huevos', así que el huevo repetido en 2 comidas era "asignado" → el
corrector lo mantuvo en ambas → la reparación era imposible por construcción y se quemó el
retry completo que el quirúrgico venía a ahorrar.

Fix: en modo RECHAZO la regla permite REDISTRIBUIR proteínas DENTRO del pool asignado del día
(mantener la proteína donde es protagonista, sustituirla en las otras comidas por OTRA opción
del pool — pasa el PROTEIN-POOL-SCRUB post-corrección). Fuera del pool sigue prohibido. El
modo markers-tras-approved conserva la regla inviolable original intacta.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_precedence_block_is_mode_conditional():
    i = _GO.index("[P1-SURGICAL-POOL-SWAP · 2026-07-05]")
    win = _GO[i:i + 2600]
    # [P1-SURGICAL-MODE-COLLISION] la condición ahora es POR DÍA (unión de modos): la regla
    # permisiva aplica solo al día que tiene issues de reject.
    assert "if _reject_mode and _reject_issues.get(day_num):" in win
    assert "_precedence_block = (" in win
    # variante reject: redistribuir dentro del pool está PERMITIDO.
    assert "SUSTITÚYELA" in win and "DEL POOL" in win
    assert "Redistribuir dentro del pool está" in win
    # variante marker: la regla inviolable original sigue intacta.
    assert "REGLA DE PRECEDENCIA INVIOLABLE" in win
    assert "MANTÉN la proteína y" in win


def test_reject_variant_still_pool_bounded():
    """La variante permisiva NO abre la puerta a proteínas fuera del pool (el scrub las
    eliminaría y quedaría un plato sin proteína)."""
    i = _GO.index("[P1-SURGICAL-POOL-SWAP · 2026-07-05]")
    win = _GO[i:i + 1800]
    assert "NUNCA" in win and "que no esté en el pool" in win


def test_prompt_interpolates_conditional_block():
    i = _GO.index("{_precedence_block}")
    # el bloque condicional reemplazó al texto hardcodeado dentro del f-string del corrector.
    win = _GO[max(0, i - 800):i]
    assert "correction_prompt" in win or "RESTRICCIONES NUTRICIONALES" in win


def test_marker_anchored_in_source():
    assert "P1-SURGICAL-POOL-SWAP" in _GO
