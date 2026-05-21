"""[P1-LOW-SIGNAL-FALLBACK · 2026-05-21] Bundle de 3 fixes relacionados con
el comportamiento del orquestador cuando el usuario tiene señal de aprendizaje
baja (chunk 2 inicial sin historial denso, score acumulado <0.40).

User request (2026-05-21):
  1. "como máximo 3 intentos, sino pudo en esos 3 intentos tiene que detenerse
     la IA y avisar" → MAX_ATTEMPTS subido 2 → 3 + flag visible al usuario.
  2. "no quiero que forze gustos que no existen" → Confidence gate sobre
     Señales 7-9 (calidad, platos recurrentes, frustración).
  3. "si el usuario no agregó suficiente información de preferencias, el chunk
     debe crear platos en base a macros + nevera/lista de compras pdf" →
     Verificación de que el fallback macro+pantry sigue funcionando cuando
     las señales se omiten.

Cambios anclados:
  - `MAX_ATTEMPTS` default 2 → 3
  - `MIN_LEARNING_CONFIDENCE` nuevo knob (default 0.40)
  - Señales 7, 8, 9 gateadas con confidence threshold
  - `plan_result._quality_degraded` flag seteado en should_retry tras max_attempts
  - Banner frontend en Dashboard.jsx lee `planData?._quality_degraded`
"""
import re
from pathlib import Path


_GRAPH_ORCH = Path(__file__).parent.parent / "graph_orchestrator.py"
_DASHBOARD_JSX = Path(__file__).parent.parent.parent / "frontend" / "src" / "pages" / "Dashboard.jsx"


# ---------------------------------------------------------------------------
# 1. MAX_ATTEMPTS subido 2 → 3
# ---------------------------------------------------------------------------

def test_max_attempts_default_is_3():
    """El default literal debe ser 3 en source. Si alguien lo baja a 2 en
    un revert, este test cae antes de regresar a la política previa."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert 'MAX_ATTEMPTS                = _env_int  ("MEALFIT_MAX_ATTEMPTS",                3)' in src, (
        "MAX_ATTEMPTS default debe ser 3 (P1-LOW-SIGNAL-FALLBACK)."
    )


def test_max_attempts_comment_explains_rationale():
    """El comentario sobre MAX_ATTEMPTS=3 debe citar P1-LOW-SIGNAL-FALLBACK y
    el porqué (notificación visible al user). Sin esto, un dev futuro que vea
    +1 attempt en latency dashboard podría revertir sin contexto."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    # Buscar el marker cerca del MAX_ATTEMPTS knob
    idx = src.find("MAX_ATTEMPTS                = _env_int  (")
    assert idx > 0
    snippet = src[max(0, idx - 800):idx + 100]
    assert "P1-LOW-SIGNAL-FALLBACK" in snippet, (
        "Comentario justificativo de MAX_ATTEMPTS=3 ausente o no cita el marker."
    )


# ---------------------------------------------------------------------------
# 2. MIN_LEARNING_CONFIDENCE knob nuevo
# ---------------------------------------------------------------------------

def test_min_learning_confidence_knob_defined():
    """El nuevo knob debe existir con default 0.40 + comentario justificativo."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert 'MIN_LEARNING_CONFIDENCE     = _env_float("MEALFIT_MIN_LEARNING_CONFIDENCE",    0.40)' in src
    # Comentario debe explicar el qué + por qué
    idx = src.find('MIN_LEARNING_CONFIDENCE     = _env_float(')
    snippet = src[max(0, idx - 1500):idx + 100]
    assert "P1-LOW-SIGNAL-FALLBACK" in snippet
    assert "macros" in snippet.lower(), (
        "Comentario debe mencionar el fallback explícito: el LLM solo recibe macros + pantry."
    )


# ---------------------------------------------------------------------------
# 3. Confidence gate sobre Señales 7-9
# ---------------------------------------------------------------------------

def test_confidence_gate_block_defined_before_senal_7():
    """El bloque `_skip_pref_signals` debe computarse ANTES de la Señal 7."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    senal_7_idx = src.find("# Señal 7: Snapshot de calidad global")
    skip_idx = src.find("_skip_pref_signals = (")
    assert senal_7_idx > 0, "Señal 7 comment not found"
    assert skip_idx > 0, "_skip_pref_signals block not found"
    assert skip_idx < senal_7_idx, (
        "_skip_pref_signals debe definirse ANTES de Señal 7. "
        f"Got: skip at {skip_idx}, senal_7 at {senal_7_idx}."
    )


def test_senal_7_uses_confidence_gate():
    """Señal 7 debe estar gateada — la condición if debe incluir
    `not _skip_pref_signals` o `_has_explicit_adherence_signal`."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    senal_7_idx = src.find("# Señal 7: Snapshot de calidad global")
    snippet = src[senal_7_idx:senal_7_idx + 800]
    assert "_skip_pref_signals" in snippet or "_has_explicit_adherence_signal" in snippet, (
        "Señal 7 debe estar wrapeada en el confidence gate."
    )


def test_senal_8_uses_confidence_gate():
    """Señal 8 (platos recurrentes) debe respetar `_skip_pref_signals`."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    senal_8_idx = src.find("# Señal 8: Platos recurrentes")
    snippet = src[senal_8_idx:senal_8_idx + 800]
    assert "_skip_pref_signals" in snippet, (
        "Señal 8 (platos recurrentes) debe estar gateada por _skip_pref_signals — "
        "sin esto seguimos forzando gustos sobre confidence baja."
    )


def test_senal_9_uses_confidence_gate():
    """Señal 9 (frustración) debe respetar `_skip_pref_signals`."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    senal_9_idx = src.find("# Señal 9: Tipos de comida que generan frustración")
    snippet = src[senal_9_idx:senal_9_idx + 800]
    assert "_skip_pref_signals" in snippet, (
        "Señal 9 (frustración) debe estar gateada por _skip_pref_signals."
    )


def test_explicit_adherence_signal_exception_documented():
    """Si _adherence_hint o _adherence_ema_hint vienen del backend con valor
    explícito (low/high/temporary_dip/drastic_change/improving), respetamos
    incluso con confidence baja — eso es señal OBSERVADA no estimada."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert "_has_explicit_adherence_signal" in src, (
        "Excepción para adherencia observada explícita debe existir — sin esto, "
        "users con patrón claro de baja-adherencia perderían la instrucción "
        "'Simplifica radicalmente' aunque su señal sea genuina."
    )


# ---------------------------------------------------------------------------
# 4. plan_result._quality_degraded flag en should_retry
# ---------------------------------------------------------------------------

def test_quality_degraded_flag_set_on_max_attempts_exit():
    """Cuando attempt >= MAX_ATTEMPTS y review_passed=False, debemos setear
    `plan_result._quality_degraded=True` ANTES de return "end" — sino el
    frontend no tiene cómo mostrar el banner."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    assert '_pr["_quality_degraded"] = True' in src, (
        "Flag `_quality_degraded=True` no se setea en plan_result tras max_attempts."
    )
    assert '_pr["_quality_degraded_reason"] = "max_attempts"' in src
    assert '_pr["_quality_degraded_attempts"]' in src


def test_quality_degraded_flag_setter_is_best_effort():
    """El setter del flag debe estar dentro de try/except — un fallo cosmético
    no debe abortar la entrega del plan (el `_emit_plan_quality_degraded_alert`
    ya documentó esa convención best-effort)."""
    src = _GRAPH_ORCH.read_text(encoding="utf-8")
    # Buscar el try que envuelve el setter
    setter_idx = src.find('_pr["_quality_degraded"] = True')
    assert setter_idx > 0
    snippet_before = src[max(0, setter_idx - 200):setter_idx]
    assert "try:" in snippet_before, (
        "El setter del flag debe estar dentro de try/except (best-effort)."
    )


# ---------------------------------------------------------------------------
# 5. Frontend banner en Dashboard.jsx
# ---------------------------------------------------------------------------

def test_dashboard_renders_banner_when_quality_degraded():
    """Dashboard.jsx debe renderizar un banner condicional cuando
    `planData?._quality_degraded` es true."""
    if not _DASHBOARD_JSX.exists():
        # En CI sin frontend checkout, skip silente. Este test SOLO falla
        # en local cuando el frontend está presente y el banner desapareció.
        import pytest
        pytest.skip("Dashboard.jsx no presente — frontend no en este checkout.")
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    assert "planData?._quality_degraded" in src, (
        "Dashboard.jsx debe leer planData?._quality_degraded para mostrar el banner."
    )
    assert "P1-LOW-SIGNAL-FALLBACK" in src, (
        "El banner debe llevar marker P1-LOW-SIGNAL-FALLBACK en comentario."
    )


def test_dashboard_banner_uses_amber_not_red():
    """El banner es informacional (plan SÍ entregado), no de error fatal.
    Debe usar paleta amber/warning, no red/error — la diferencia importa para
    el user que entiende que tiene un plan funcional pero subóptimo."""
    if not _DASHBOARD_JSX.exists():
        import pytest
        pytest.skip("Dashboard.jsx no presente.")
    src = _DASHBOARD_JSX.read_text(encoding="utf-8")
    # Buscar el bloque del banner por su comment marker
    banner_idx = src.find("[P1-LOW-SIGNAL-FALLBACK · 2026-05-21] Banner")
    assert banner_idx > 0
    snippet = src[banner_idx:banner_idx + 2000]
    # Color amber/warning detectable por presencia de tonos amarillos/ámbar
    assert "#FFFBEB" in snippet or "#FEF3C7" in snippet or "#FCD34D" in snippet, (
        "Banner debe usar paleta amber (#FFFBEB/#FEF3C7/#FCD34D), no red."
    )
    # El copy debe mencionar "Cambiar Plato" como acción remediadora
    assert "Cambiar Plato" in snippet, (
        "Banner debe mencionar 'Cambiar Plato' como acción remediadora — "
        "el usuario necesita saber qué hacer cuando el plan está degradado."
    )
