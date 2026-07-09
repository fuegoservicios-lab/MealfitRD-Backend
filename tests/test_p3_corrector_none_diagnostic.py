"""[P3-CORRECTOR-NONE-DIAGNOSTIC · 2026-07-09] Cuando el corrector quirúrgico Pro
(`_attempt_pro_critique_correction`) devuelve None SIN lanzar excepción, el structured-output no
produjo un `SingleDayPlanModel` válido (vacío / no-parseable / refusal) — hoy es un PUNTO CIEGO: el
log dice "retornó None" pero no POR QUÉ. Este knob (default OFF, para no pagar una 2ª llamada en prod)
re-invoca el modelo RAW (sin structured output) con el mismo prompt y loguea content + finish_reason →
revela el patrón para tunear prompt/modelo. Cero-ripple: no toca el success-path ni el emit de costo.

Forense: corrida corr=8b721589 (2026-07-09) — Pro retornó None para Día 3 en el retry quirúrgico →
correción perdida → whack-a-mole. Sin este diagnóstico no se puede saber si fue prosa, JSON malformado,
vacío o refusal.
"""
import os

import graph_orchestrator as go

_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(*parts):
    with open(os.path.join(*parts), encoding="utf-8") as f:
        return f.read()


def _src():
    return _read(_BACKEND, "graph_orchestrator.py")


def test_knob_defined_default_off():
    assert go.CORRECTOR_NONE_DIAGNOSTIC_ENABLED is False, "el diagnóstico debe nacer OFF (2ª llamada = costo)"
    assert "MEALFIT_CORRECTOR_NONE_DIAGNOSTIC" in _src()


def test_marker_present():
    assert "P3-CORRECTOR-NONE-DIAGNOSTIC" in _src()


def test_diagnostic_block_inside_pro_corrector_before_return_none():
    """El bloque de diagnóstico debe vivir en _attempt_pro_critique_correction, gated por el knob, y
    ANTES del `return None, "pro_returned_none"` (para capturar el raw del caso que falló)."""
    src = _src()
    fn = src[src.find("async def _attempt_pro_critique_correction"):]
    fn = fn[: fn.find("\nasync def ", 1) if fn.find("\nasync def ", 1) > 0 else 6000]
    pos_gate = fn.find("if CORRECTOR_NONE_DIAGNOSTIC_ENABLED")
    pos_ret = fn.find('return None, "pro_returned_none"')
    assert pos_gate >= 0, "falta el bloque diagnóstico gated por CORRECTOR_NONE_DIAGNOSTIC_ENABLED"
    assert pos_ret >= 0, "no se encontró el return pro_returned_none"
    assert pos_gate < pos_ret, "el diagnóstico debe ir ANTES del return None (para capturar el raw fallido)"


def test_diagnostic_captures_finish_reason_and_content():
    src = _src()
    fn = src[src.find("if CORRECTOR_NONE_DIAGNOSTIC_ENABLED"):]
    fn = fn[:1500]
    assert "finish_reason" in fn, "el diagnóstico debe loguear finish_reason del raw"
    assert "content" in fn, "el diagnóstico debe loguear el content del raw"
    assert "CORRECTOR-NONE-DIAGNOSTIC" in fn, "log anclado al marker"


def test_diagnostic_is_failsafe():
    """El re-invoke diagnóstico NUNCA debe romper el path del corrector (try/except propio)."""
    src = _src()
    fn = src[src.find("if CORRECTOR_NONE_DIAGNOSTIC_ENABLED"):]
    fn = fn[:1500]
    assert "except Exception" in fn, "el diagnóstico debe ser fail-safe (except propio)"
