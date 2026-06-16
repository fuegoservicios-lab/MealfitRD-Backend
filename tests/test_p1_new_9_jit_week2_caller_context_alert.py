"""[P1-NEW-9 · 2026-05-11] JIT week-2 (proactive_agent) debe correlacionar
el `plan_quality_degraded` alert al plan real, no a `no_plan_id`.

Bug original (audit 2026-05-11):
    `_trigger_week2_background_generation` (proactive_agent.py) llamaba
    `run_plan_pipeline(form_data, ...)` sin atribución del plan target.
    Cuando `should_retry` decidía "end" con `review_passed=False` (5
    ramas: critical/high_contextual/max_attempts/invalid_pipeline_start/
    budget_exhausted), el alert se emitía pero `plan_result.id` aún
    no existía (la extensión week-2 NO inserta plan nuevo, solo extiende
    el existente vía `update_meal_plan_data`). Resultado:
        alert_key = "plan_quality_degraded:<user_id>:no_plan_id"
    SRE veía la alerta pero NO sabía qué plan se entregó degradado, y
    los alerts de week-2 colapsaban con los de `/generate-plan` inicial
    para el mismo usuario (mismo alert_key).

Fix:
    1. JIT caller inyecta `form_data["_caller_target_plan_id"] = plan_id`
       y `form_data["_caller_context"] = "jit_week2"` ANTES del pipeline.
    2. `_TRUSTED_INTERNAL_FORM_KEYS` incluye ambas keys (sino el strip
       defensive de routers/orchestrator las eliminaría).
    3. `_emit_plan_quality_degraded_alert` lee `_caller_target_plan_id`
       como fallback antes de `"no_plan_id"`, Y añade `caller_context`
       al metadata.

Estrategia del test (parser-based sobre el código fuente):
    1. Whitelist incluye ambas keys.
    2. Helper helper resuelve plan_id desde `_caller_target_plan_id`.
    3. Helper escribe `caller_context` en metadata.
    4. `_trigger_week2_background_generation._bg_task` setea ambas keys
       en form_data ANTES de `run_plan_pipeline(...)`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ORCH_FP = _REPO_ROOT / "backend" / "graph_orchestrator.py"
_PROACT_FP = _REPO_ROOT / "backend" / "proactive_agent.py"


@pytest.fixture(scope="module")
def orch_src() -> str:
    return _ORCH_FP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def proact_src() -> str:
    return _PROACT_FP.read_text(encoding="utf-8")


def test_whitelist_includes_caller_keys(orch_src: str):
    """`_TRUSTED_INTERNAL_FORM_KEYS` debe incluir ambas keys.

    Sin esto, `_strip_untrusted_internal_keys` las eliminaría al
    comienzo del pipeline y el alert volvería a usar `no_plan_id`.
    """
    decl_match = re.search(
        r"_TRUSTED_INTERNAL_FORM_KEYS\s*:\s*frozenset\s*=\s*frozenset\(\{([\s\S]+?)\}\)",
        orch_src,
    )
    assert decl_match, "_TRUSTED_INTERNAL_FORM_KEYS no declarada como frozenset({...})"
    body = decl_match.group(1)
    for key in ('"_caller_target_plan_id"', '"_caller_context"'):
        assert key in body, (
            f"P1-NEW-9 regresión: {key} no está en _TRUSTED_INTERNAL_FORM_KEYS. "
            f"`_strip_untrusted_internal_keys` la eliminará al inicio del "
            f"pipeline → el alert volverá a colapsar a 'no_plan_id'."
        )


def test_alert_helper_reads_caller_target_plan_id(orch_src: str):
    """`_emit_plan_quality_degraded_alert` debe leer
    `form_data['_caller_target_plan_id']` como fallback antes de
    `"no_plan_id"`."""
    helper_start = orch_src.find("def _emit_plan_quality_degraded_alert(")
    assert helper_start > 0, "_emit_plan_quality_degraded_alert no encontrado"
    helper_end = orch_src.find("\ndef should_retry(", helper_start)
    assert helper_end > helper_start, "fin del helper no encontrado"
    body = orch_src[helper_start:helper_end]

    # El fallback debe existir y aparecer ANTES de `"no_plan_id"`.
    fallback_match = re.search(
        r'form_data\.get\(\s*[\'"]_caller_target_plan_id[\'"]\s*\)',
        body,
    )
    assert fallback_match, (
        "P1-NEW-9 regresión: `_emit_plan_quality_degraded_alert` ya no "
        "lee `form_data['_caller_target_plan_id']` como fallback. Sin "
        "este fallback, el alert para JIT week-2 vuelve a usar "
        '"no_plan_id" y SRE no puede correlacionar.'
    )
    noplan_idx = body.find('"no_plan_id"')
    assert noplan_idx > fallback_match.start(), (
        "P1-NEW-9 regresión: el orden es incorrecto. "
        "`_caller_target_plan_id` debe consultarse ANTES de "
        '`"no_plan_id"` para que tenga efecto.'
    )


def test_alert_metadata_includes_caller_context(orch_src: str):
    """El metadata del alert debe incluir `caller_context`."""
    helper_start = orch_src.find("def _emit_plan_quality_degraded_alert(")
    helper_end = orch_src.find("\ndef should_retry(", helper_start)
    body = orch_src[helper_start:helper_end]
    assert re.search(r'[\'"]caller_context[\'"]\s*:\s*caller_context', body), (
        "P1-NEW-9 regresión: el dict `metadata` ya no incluye "
        "`'caller_context': caller_context`. Sin ese campo, SRE no puede "
        "filtrar las alerts por origen (initial_generate vs jit_week2)."
    )
    # Y default a 'initial_generate' cuando form_data no lo trae.
    assert re.search(
        r'caller_context\s*=\s*form_data\.get\(\s*[\'"]_caller_context[\'"]\s*\)'
        r'\s*or\s*[\'"]initial_generate[\'"]',
        body,
    ), (
        "P1-NEW-9 regresión: el default de caller_context no es "
        "'initial_generate'. Sin default, alerts viejos podrían quedar con "
        "None y romper filtros downstream."
    )


def test_jit_week2_sets_caller_keys_before_pipeline(proact_src: str):
    """`_bg_task` debe setear ambas keys en `form_data` ANTES de
    invocar `run_plan_pipeline`."""
    bg_match = re.search(
        r"def\s+_bg_task\s*\(\s*\)\s*:",
        proact_src,
    )
    assert bg_match, "`_bg_task` no encontrado en proactive_agent.py"

    # Boundary: el primer `run_plan_pipeline(` después de `_bg_task`.
    pipeline_idx = proact_src.find("run_plan_pipeline(", bg_match.end())
    assert pipeline_idx > 0, "run_plan_pipeline(...) call no encontrado"
    pre = proact_src[bg_match.end():pipeline_idx]

    assert re.search(
        r'form_data\[\s*[\'"]_caller_target_plan_id[\'"]\s*\]\s*=\s*plan_id',
        pre,
    ), (
        "P1-NEW-9 regresión: `_bg_task` ya no setea "
        "`form_data['_caller_target_plan_id'] = plan_id` ANTES de "
        "`run_plan_pipeline(...)`. Sin esa inyección, el alert "
        "colapsará al pattern `<user_id>:no_plan_id`."
    )
    assert re.search(
        r'form_data\[\s*[\'"]_caller_context[\'"]\s*\]\s*=\s*[\'"]jit_week2[\'"]',
        pre,
    ), (
        "P1-NEW-9 regresión: `_bg_task` ya no setea "
        "`form_data['_caller_context'] = 'jit_week2'` ANTES de "
        "`run_plan_pipeline(...)`. Sin esa tag, SRE no podrá filtrar "
        "alerts por origen JIT week-2."
    )


def test_claude_md_documents_caller_context(orch_src):
    """La tabla canónica de alerts debe mencionar `caller_context`
    en la fila `plan_quality_degraded` — sin docs, una regresión que
    elimine el campo del metadata podría pasar code review.

    [stale-parser fix] La tabla canónica de ~32 `alert_key` se movió de
    CLAUDE.md a `backend/docs/system_alerts_resolution_table.md` (P2-NEW-3
    + convención doc-first del repo: CLAUDE.md queda con header + link, el
    detalle vive en `docs/`). La fila `plan_quality_degraded` con el marker
    P1-NEW-9 y `caller_context` ahora vive ahí. Este test apunta al doc
    canónico actual."""
    alerts_doc = (
        _REPO_ROOT / "backend" / "docs" / "system_alerts_resolution_table.md"
    ).read_text(encoding="utf-8")
    # La fila plan_quality_degraded debe mencionar P1-NEW-9 y caller_context.
    row_idx = alerts_doc.find("plan_quality_degraded:<user_id>:<plan_id>")
    assert row_idx > 0, (
        "fila `plan_quality_degraded` no encontrada en "
        "backend/docs/system_alerts_resolution_table.md"
    )
    # Ventana razonable para leer hasta el final de la fila.
    row_chunk = alerts_doc[row_idx:row_idx + 1500]
    assert "P1-NEW-9" in row_chunk, (
        "P1-NEW-9 regresión: la fila `plan_quality_degraded` en "
        "system_alerts_resolution_table.md ya no menciona el P-fix. Sin "
        "marker, un revisor futuro no encontrará el contexto de por qué "
        "existe `caller_context`."
    )
    assert "caller_context" in row_chunk, (
        "P1-NEW-9 regresión: la fila `plan_quality_degraded` no menciona "
        "`caller_context`. Sin esa mención, alguien que vea el field en "
        "metadata sin contexto puede pensar que es ruido y removerlo."
    )
