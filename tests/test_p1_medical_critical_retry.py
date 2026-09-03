# [P1-MEDICAL-CRITICAL-RETRY · 2026-08-09] Un critical MÉDICO del reviewer va hoy DIRECTO al
# fallback matemático (plan muerto, sin intento de corrección) mientras dieta/alérgenos ya tienen
# UN retry informado (P1-DAYGEN-DIET-CONVERGE). La clase medida en la corrida 31304538636
# («no especifica descremado/bajo en sodio/porción», 10/20 perfiles al fallback) es corregible
# con la directiva + las notas deterministas que el retry ya encuentra puestas
# (P1-CONDITION-SAFETY-NOTES). NO debilita el guard: severity sigue critical, el reviewer
# re-gatea con los mismos escáneres, y la reincidencia (attempt>1) aborta al fallback terminal
# idéntico al de hoy.
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_SRC = open(os.path.join(os.path.dirname(__file__), "..", "graph_orchestrator.py"),
            encoding="utf-8").read()


def _critical_block():
    i = _SRC.index('if severity == "critical":')
    j = _SRC.index('_emit_plan_quality_degraded_alert(state, exit_reason="critical"', i)
    return _SRC[i:j]


def test_medical_critical_obtiene_retry():
    blk = _critical_block()
    assert "MEDICAL_CRITICAL_REGEN_ENABLED and not _diet_allergen_critical" in blk, (
        "el critical médico (no dieta/alérgeno) debe entrar al MISMO gate de un-retry-informado; "
        "sin esto, 10/20 perfiles de la corrida fueron al fallback sin intento de corrección"
    )
    assert "P1-MEDICAL-CRITICAL-RETRY" in blk


def test_gate_sigue_acotado():
    blk = _critical_block()
    # attempt==1 + presupuesto: sin esto sería un loop, no un retry.
    assert 'int(state.get("attempt", 1)) == 1' in blk
    assert "MIN_RETRY_BUDGET_S" in blk


def test_knob_existe_con_default_on():
    assert 'MEDICAL_CRITICAL_REGEN_ENABLED = _env_bool("MEALFIT_MEDICAL_CRITICAL_REGEN", True)' in _SRC


def test_fallback_terminal_intacto():
    # la reincidencia sigue abortando: el emit + return "end" viven tras el gate.
    i = _SRC.index('if severity == "critical":')
    tail = _SRC[i:i + 4000]
    assert 'exit_reason="critical"' in tail and '"🚨 [ORQUESTADOR] Rechazo CRÍTICO' in tail
