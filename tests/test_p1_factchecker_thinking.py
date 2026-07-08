"""[P1-FACTCHECKER-THINKING · 2026-07-08] Thinking mode V4 en la 3ra superficie de
juicio clínico: el fact-checker (FASE 1) que investiga alergias/condiciones/reacciones
cruzadas y sintetiza el REPORTE que alimenta al reviewer médico.

Contexto: sesión de A/B de thinking sobre 3 superficies. Regla medida:
  - Reviewer médico (output chico = veredicto)   → thinking `max` MEJORA (estratifica riesgo).
  - Fact-checker  (output chico = reporte)        → thinking `high` MEJORA SUSTANCIAL
    (A/B warfarina+mariscos: OFF=34s/2.6k chars; HIGH=53s/3.2k chars atrapó ADEMÁS la
    interacción fibra↔absorción de warfarina + CYP450 + reactividad cruzada sistemática).
    `max` (72s) NO fue estrictamente mejor que `high` → high es el sweet spot.
  - Corrector quirúrgico (output GRANDE = día completo) → thinking REVIENTA el timeout.

Diferencia técnica clave: el fact-checker usa `bind_tools` SIN `tool_choice` forzado →
thinking NATIVO (no requiere el workaround json_mode del reviewer/surgical). El cap de
30s/iter es muy justo para reasoning → se sube en la rama thinking vía knob dedicado.
Nace OFF (convención medir→actuar). Fail-safe: si el API rechaza el campo, el loop cae
al reporte precautorio estándar (mismo contrato de hoy).
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND = os.path.dirname(_HERE)


def _read(rel):
    with open(os.path.join(_BACKEND, rel), encoding="utf-8") as f:
        return f.read()


_GO = _read("graph_orchestrator.py")


# ---------------------------------------------------------------------------
# knobs: nacen OFF
# ---------------------------------------------------------------------------

def test_factchecker_knobs_born_off():
    assert '_env_bool("MEALFIT_FACT_CHECKER_THINKING", False)' in _GO
    assert '_env_str("MEALFIT_FACT_CHECKER_THINKING_EFFORT", "")' in _GO
    assert '_env_int("MEALFIT_FACT_CHECKER_THINKING_TIMEOUT_S", 60' in _GO


# ---------------------------------------------------------------------------
# rama thinking: bind_tools nativo (SIN json_mode)
# ---------------------------------------------------------------------------

def test_factchecker_thinking_branch():
    i = _GO.index("_fc_thinking = FACT_CHECKER_THINKING_ENABLED")
    win = _GO[i:i + 1100]
    assert '_fc_think_body = {"type": "enabled"}' in win
    assert 'extra_body={"thinking": _fc_think_body}' in win
    assert 'bind_tools([consultar_base_datos_medica])' in win
    # thinking nativo: NO usa el workaround json_mode del reviewer/surgical.
    assert 'json_mode' not in win, \
        "el fact-checker usa bind_tools sin tool_choice forzado → thinking nativo"
    # rama estándar intacta (sin extra_body).
    assert ".bind_tools([consultar_base_datos_medica])\n" in _GO[i:i + 1600]


def test_factchecker_thinking_iter_timeout_bumped():
    """El cap de 30s/iter es muy justo para reasoning → sube al knob dedicado SOLO
    en la rama thinking; la rama estándar conserva 30s."""
    assert "_fc_iter_timeout = float(FACT_CHECKER_THINKING_TIMEOUT_S) if _fc_thinking else 30.0" in _GO
    assert "timeout=_fc_iter_timeout" in _GO


def test_factchecker_gated_to_medical_flags():
    """La FASE 1 (y por ende el thinking) solo corre con flags médicos reales."""
    assert "if _has_real_medical_flags(allergies) or _has_real_medical_flags(medical_conditions):" in _GO


def test_marker_anchored_in_source():
    assert "P1-FACTCHECKER-THINKING" in _GO
