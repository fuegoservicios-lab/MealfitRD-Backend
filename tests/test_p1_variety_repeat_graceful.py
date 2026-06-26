"""[P1-VARIETY-REPEAT-GRACEFUL · 2026-06-26] Degradación con gracia del gate de variedad-repetida.

INCIDENTE RAÍZ (user d4bc3af5, corr=20394088, 2026-06-26 06:19): una renovación falló con
"La IA está temporalmente saturada y no pudimos generar tu plan" pese a que DeepSeek estaba sano
(días 1 y 2 se corrigieron con el modelo Pro). Causa: el gate `P2-VARIETY-GATE-REPEAT` (fruta dulce
repetida el mismo día) rechazó los 3 intentos con severity='high' → review nunca pasó → ruta de
agotamiento → `_repair_partial_plan` marcó `_is_fallback=True` → el FALLBACK-GUARD/SSE rehúsa
persistir y emite el mensaje genérico "saturada" (ENGAÑOSO) → el usuario se queda SIN plan.

FIX: en el intento FINAL (attempt >= MAX_ATTEMPTS) el gate de variedad-repetida es ADVISORY — NO
pone approved=False. Una fruta repetida es cosmética y jamás debe convertir un plan válido en
cero-plan. En intentos 1..N-1 SÍ rechaza (preserva la presión de diversificación). Knob:
MEALFIT_VARIETY_REPEAT_GATE_LAST_ATTEMPT_ADVISORY (default True). Anchor: P1-VARIETY-REPEAT-GRACEFUL.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import graph_orchestrator as go


_SRC = open(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "graph_orchestrator.py"),
    encoding="utf-8",
).read()


def test_knob_existe_y_default_on():
    """El knob de degradación con gracia existe y arranca ON (default seguro)."""
    assert go.VARIETY_REPEAT_GATE_LAST_ATTEMPT_ADVISORY is True


def test_knob_registrado():
    from knobs import get_knobs_registry_snapshot
    snap = get_knobs_registry_snapshot()
    assert "MEALFIT_VARIETY_REPEAT_GATE_LAST_ATTEMPT_ADVISORY" in snap


def test_marker_presente():
    assert "P1-VARIETY-REPEAT-GRACEFUL" in _SRC


def _gate_block() -> str:
    """Aísla el bloque de cableado del gate de variedad-repetida dentro de review_plan_node."""
    start = _SRC.index("_rep_issues = _variety_repeat_gate_issues(_vr)")
    # El bloque termina en el except del try VARIETY_HARD_GATE.
    end = _SRC.index("except Exception as _vg_e", start)
    return _SRC[start:end]


def test_gate_es_attempt_aware_en_intento_final():
    """Parser-based: en el intento final, el gate consulta MAX_ATTEMPTS y el knob, y la rama
    advisory aparece ANTES del rechazo (approved=False) — el rechazo vive en el `else`."""
    block = _gate_block()
    # Consulta el número de intento y el tope.
    assert "_vg_is_final" in block
    assert "MAX_ATTEMPTS" in block
    assert 'state.get("attempt"' in block
    # La condición de degradación con gracia está presente.
    assert "VARIETY_REPEAT_GATE_LAST_ATTEMPT_ADVISORY and _vg_is_final" in block
    # El rechazo (approved=False por fruta repetida) está DESPUÉS del branch advisory → solo
    # se alcanza cuando NO es el intento final (o el knob está OFF).
    idx_advisory = block.index("VARIETY_REPEAT_GATE_LAST_ATTEMPT_ADVISORY and _vg_is_final")
    idx_else = block.index("else:", idx_advisory)
    idx_reject = block.index("approved = False", idx_advisory)
    assert idx_advisory < idx_else < idx_reject, (
        "el rechazo del gate de variedad-repetida debe estar en el `else` del check de intento final"
    )


def test_no_rechazo_directo_fuera_del_else():
    """No debe quedar un `approved = False` por fruta-repetida que se ejecute incondicionalmente
    (regresión: el bug original rechazaba en TODOS los intentos)."""
    block = _gate_block()
    # El único approved=False del bloque debe estar dentro del for del else (indentado profundo).
    # Heurística robusta: cada 'approved = False' del bloque va precedido, en su misma rama, por
    # el `for _rep_issue in _rep_issues:` del else.
    assert "for _rep_issue in _rep_issues:" in block
    # El for del else viene después de la rama advisory.
    assert block.index("VARIETY_REPEAT_GATE_LAST_ATTEMPT_ADVISORY") < block.index("for _rep_issue in _rep_issues:")


def test_helper_de_issues_intacto():
    """El helper puro sigue detectando fruta repetida (el fix NO toca la detección, solo el
    momento de rechazar). Garantiza que en intentos tempranos el gate aún tiene con qué rechazar."""
    issues = go._variety_repeat_gate_issues({"fruit_repeats": 1, "same_day_repeats": 0})
    assert len(issues) == 1 and "FRUTA REPETIDA" in issues[0]


def test_knob_off_revierte_a_rechazo_siempre():
    """Con el knob OFF, el código cae al `else` (rechazo) en todos los intentos — rollback exacto
    al comportamiento previo. Verificado a nivel de fuente: el rechazo no depende solo de _vg_is_final."""
    block = _gate_block()
    # La guarda usa AND entre el knob y _vg_is_final → si el knob es False, va al else (rechazo).
    assert re.search(r"if\s+VARIETY_REPEAT_GATE_LAST_ATTEMPT_ADVISORY\s+and\s+_vg_is_final\s*:", block)
