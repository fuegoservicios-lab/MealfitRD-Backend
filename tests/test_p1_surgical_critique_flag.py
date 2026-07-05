"""[P1-SURGICAL-CRITIQUE-FLAG · 2026-07-05] Paridad del flag `_critique_applied` en el corrector.

Caso vivo corr=214635d9: el 🩹 reparó el Día 1 vía corrector FLASH exitoso, pero el path flash
de `_re_correct_one` NO seteaba `_critique_applied=True` (el pro-fallback SÍ, desde
P-PRO-FALLBACK-ON-NONE). El skeleton-fidelity aplica threshold ESTRICTO (2 missing) a días sin
el flag → "Día 1 omitió [conejo, huevos]" mató una reparación quirúrgica válida (el corrector
había swapeado la proteína repetida legítimamente) y quemó el retry completo que el 🩹 ahorraba.

Semántica ya documentada (P3-SKELETON-FIDELITY-CRITIQUE-AWARE): día reescrito por el corrector
deviene legítimamente del skeleton → threshold relajado a 3 (solo flagea si removió TODO).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)

with open(os.path.join(_BACKEND, "graph_orchestrator.py"), encoding="utf-8") as f:
    _GO = f.read()


def test_flash_path_sets_critique_applied():
    i = _GO.index("[P5-MARKER-REGEN] Día {day_num} re-corregido exitosamente")
    win = _GO[max(0, i - 1400):i]
    assert 'corrected_day["_critique_applied"] = True' in win, \
        "el path FLASH del corrector quirúrgico debe marcar el día como critique-corrected"


def test_pro_fallback_keeps_parity():
    i = _GO.index('corrected["_critique_applied"] = True')
    win = _GO[max(0, i - 1200):i + 200]
    assert "PRO-FALLBACK" in win or "pro" in win.lower(), \
        "el pro-fallback conserva su flag (paridad bidireccional)"


def test_fidelity_semantics_documented():
    assert "_missing_threshold = 3 if _critique_applied_for_day else 2" in _GO, \
        "el threshold relajado para días critique-corrected es el contrato que este flag activa"


def test_marker_anchored_in_source():
    assert "P1-SURGICAL-CRITIQUE-FLAG" in _GO
