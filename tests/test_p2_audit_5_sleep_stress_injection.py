"""[P2-AUDIT-5 · 2026-05-10] Verifica que `sleepHours` y `stressLevel` —
campos validados como required en `_REQUIRED_FORM_FIELDS` — son consumidos
por el pipeline vía `build_sleep_stress_context` e inyectados al planner +
day generator.

Bug original (audit 2026-05-10):
  Ambos campos vivían en `_REQUIRED_FORM_FIELDS` (routers/plans.py:306) con
  enum validation estricto (routers/plans.py:447-448) pero el grep
  cross-codebase confirmaba CERO consumers en graph_orchestrator/prompts/
  agent/ai_helpers. Comment en routers/plans.py:417 afirmaba "solo hints
  textuales al LLM" — claim falso. El usuario completaba dos pasos del
  wizard (QSleep, QStress) sin que afectara su plan.

Fix:
  1. `prompts/plan_generator.py:build_sleep_stress_context(form_data)` —
     helper que retorna bloque markdown con hints per-valor accionable.
  2. `graph_orchestrator.py` importa + usa el helper en el dict `ctx`
     (key `sleep_stress_context`) y lo cabla en el prompt del planner
     (alrededor de L3605) y del day generator (alrededor de L3955).
  3. `routers/plans.py:417` comment actualizado para no mentir.

Tests:
  - Helper produce hint correcto para cada valor accionable.
  - Helper retorna "" si todos los valores son no-accionables (7-8h sleep
    + Bajo|Moderado stress) o están ausentes.
  - Drift detection parser-based: cada field en `_REQUIRED_FORM_FIELDS`
    DEBE tener ≥1 consumer en backend (anchor que prevenga futuros orphan
    fields).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND = _REPO_ROOT / "backend"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Helper unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_build_sleep_stress_context_low_sleep_only():
    """Sueño <6h emite hint, estrés Bajo no — bloque emitido con solo
    el hint de sueño."""
    from prompts.plan_generator import build_sleep_stress_context
    out = build_sleep_stress_context({"sleepHours": "< 6 horas", "stressLevel": "Bajo"})
    assert out.strip(), "Debe emitir bloque cuando hay al menos un hint accionable"
    assert "Sueño reportado" in out
    assert "< 6 horas" in out
    assert "Estrés reportado" not in out, "Estrés Bajo no es accionable, no debe aparecer"


def test_build_sleep_stress_context_high_stress_only():
    """Estrés Alto emite hint, sueño 7-8h no — bloque emitido con solo
    el hint de estrés."""
    from prompts.plan_generator import build_sleep_stress_context
    out = build_sleep_stress_context({"sleepHours": "7-8 horas", "stressLevel": "Alto"})
    assert out.strip(), "Debe emitir bloque cuando hay al menos un hint accionable"
    assert "Estrés reportado" in out
    assert "Alto" in out
    assert "magnesio" in out
    assert "Sueño reportado" not in out, "Sueño 7-8h no es accionable, no debe aparecer"


def test_build_sleep_stress_context_both_actionable():
    """Sueño <6h + estrés Muy Alto emiten ambos hints."""
    from prompts.plan_generator import build_sleep_stress_context
    out = build_sleep_stress_context({
        "sleepHours": "< 6 horas",
        "stressLevel": "Muy Alto",
    })
    assert "Sueño reportado" in out
    assert "Estrés reportado" in out
    assert "Muy Alto" in out
    assert "magnesio" in out


def test_build_sleep_stress_context_all_non_actionable_returns_empty():
    """Sueño 7-8h + estrés Bajo → "" (no signal accionable)."""
    from prompts.plan_generator import build_sleep_stress_context
    out = build_sleep_stress_context({"sleepHours": "7-8 horas", "stressLevel": "Bajo"})
    assert out == "", f"Esperaba string vacío, recibí: {out!r}"


def test_build_sleep_stress_context_empty_input():
    """form_data vacío → ""."""
    from prompts.plan_generator import build_sleep_stress_context
    assert build_sleep_stress_context({}) == ""
    assert build_sleep_stress_context({"sleepHours": "", "stressLevel": ""}) == ""


def test_build_sleep_stress_context_non_dict_input():
    """None/list/string como form_data → "" (no crash)."""
    from prompts.plan_generator import build_sleep_stress_context
    assert build_sleep_stress_context(None) == ""  # type: ignore[arg-type]
    assert build_sleep_stress_context([]) == ""  # type: ignore[arg-type]
    assert build_sleep_stress_context("not a dict") == ""  # type: ignore[arg-type]


def test_build_sleep_stress_context_unknown_value_silent():
    """Valor fuera del enum (ej. legacy) → no crash; ignora ese campo."""
    from prompts.plan_generator import build_sleep_stress_context
    # "Extremo" no está en _STRESS_LEVEL_ENUM; debe ser ignorado.
    out = build_sleep_stress_context({"sleepHours": "< 6 horas", "stressLevel": "Extremo"})
    assert "Sueño reportado" in out
    assert "Estrés reportado" not in out


# ─────────────────────────────────────────────────────────────────────────────
# 2. Wiring tests — el helper está importado y cableado en graph_orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def test_helper_imported_in_graph_orchestrator():
    """`graph_orchestrator.py` debe importar `build_sleep_stress_context`
    desde prompts.plan_generator (al lado de build_motivation_context)."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "build_sleep_stress_context" in src, (
        "graph_orchestrator.py no importa `build_sleep_stress_context`. "
        "Sin el import, el ctx dict no puede llamarlo y el campo del wizard "
        "queda orphan otra vez."
    )


def test_helper_called_in_ctx_dict():
    """El helper debe ser invocado y asignado a `sleep_stress_context` en
    el dict de context del orquestador."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    pattern = re.compile(
        r'["\']sleep_stress_context["\']\s*:\s*build_sleep_stress_context\s*\(',
    )
    assert pattern.search(src), (
        "No se encontró asignación `'sleep_stress_context': "
        "build_sleep_stress_context(...)` en graph_orchestrator.py. "
        "Sin esta key el ctx['sleep_stress_context'] sería KeyError."
    )


def test_planner_prompt_includes_sleep_stress_context():
    """El f-string del planner prompt debe interpolar `ctx['sleep_stress_context']`."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    assert "ctx['sleep_stress_context']" in src, (
        "El planner prompt no incluye `ctx['sleep_stress_context']`. "
        "Sin esa interpolación el LLM no recibe la señal."
    )


def test_day_generator_and_planner_both_include_sleep_stress():
    """Tanto planner como day generator deben interpolar el bloque.
    Mínimo 2 ocurrencias de `ctx['sleep_stress_context']` en
    graph_orchestrator.py (espejo del patrón de motivation_context)."""
    src = (_BACKEND / "graph_orchestrator.py").read_text(encoding="utf-8")
    count = src.count("ctx['sleep_stress_context']")
    assert count >= 2, (
        f"Esperaba al menos 2 interpolaciones de ctx['sleep_stress_context'] "
        f"(planner + day generator), encontré {count}. Si removiste alguna, "
        f"el sleep/stress hint solo llega a uno de los dos LLM calls."
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Drift detection: required form fields no orphan
# ─────────────────────────────────────────────────────────────────────────────

def _extract_required_form_fields() -> list[str]:
    """Parsea `_REQUIRED_FORM_FIELDS` desde routers/plans.py.

    Captura los strings dentro de la tupla. Acepta multiline / comments."""
    src = (_BACKEND / "routers" / "plans.py").read_text(encoding="utf-8")
    # Encuentra la asignación.
    m = re.search(
        r"_REQUIRED_FORM_FIELDS\s*=\s*\((.*?)\)\s*\n",
        src,
        re.DOTALL,
    )
    assert m, "No se encontró `_REQUIRED_FORM_FIELDS = (...)` en routers/plans.py"
    body = m.group(1)
    # Remueve comments inline.
    body_clean = re.sub(r"#[^\n]*\n", "\n", body)
    # Extrae todos los string literals.
    fields = re.findall(r'["\']([^"\']+)["\']', body_clean)
    return fields


def _field_has_consumer(field_name: str) -> bool:
    """Busca el field en los archivos donde un consumer downstream razonablemente
    lo leería: graph_orchestrator, agent, ai_helpers, prompts/, cron_tasks."""
    targets = (
        _BACKEND / "graph_orchestrator.py",
        _BACKEND / "agent.py",
        _BACKEND / "ai_helpers.py",
        _BACKEND / "cron_tasks.py",
        _BACKEND / "proactive_agent.py",
        _BACKEND / "tools.py",
        _BACKEND / "prompts" / "plan_generator.py",
        _BACKEND / "prompts" / "chat_agent.py",
        _BACKEND / "prompts" / "memory.py",
        _BACKEND / "nutrition_calculator.py",
        _BACKEND / "db_profiles.py",
        _BACKEND / "routers" / "diary.py",
    )
    for fp in targets:
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8")
        # Heurística: el field aparece en el archivo como literal (probable
        # consumer). `routers/plans.py` es excluido — esos sites son validación,
        # no consumption real.
        if field_name in text:
            return True
    return False


def test_every_required_form_field_has_consumer():
    """Cada field en `_REQUIRED_FORM_FIELDS` DEBE tener ≥1 consumer en backend.

    Esta es la anti-regresión de P2-AUDIT-5: si alguien añade un required
    field al wizard sin cablear un consumer, este test falla con copy
    explicativo. Evita repetir el bug `sleepHours/stressLevel` orphan.
    """
    required = _extract_required_form_fields()
    assert required, "Esperaba al menos 1 field en `_REQUIRED_FORM_FIELDS`"

    orphan_fields = [
        field for field in required if not _field_has_consumer(field)
    ]
    if orphan_fields:
        pytest.fail(
            "Los siguientes campos están en `_REQUIRED_FORM_FIELDS` "
            "(rechazo 422 si ausente) pero NO tienen consumer en backend:\n"
            "  - " + "\n  - ".join(orphan_fields) + "\n\n"
            "El wizard pide el dato al usuario pero el pipeline lo descarta. "
            "Opciones:\n"
            "  1. Inyectar al prompt LLM como hint (ver "
            "`build_motivation_context` o `build_sleep_stress_context` como "
            "patrón).\n"
            "  2. Si no se va a usar todavía, quitar de `_REQUIRED_FORM_FIELDS` "
            "y comentar 'captured for future model'.\n"
            "  3. Eliminar el paso del wizard.\n\n"
            "Si añadiste consumer en un archivo no listado en "
            "`_field_has_consumer`, actualizar este test."
        )
