"""[P3-CLEANUP · 2026-05-11] Anchor + tests del P-fix de cleanup:

  A) SOP `deploy_lag_drift_vs_expected` documentado en CLAUDE.md.
  B) Sweep adicional con TTL agresivo para alerts `scheduler_*` con
     job_id UUID hex one-off (32 caracteres a-f0-9).

Bug original:
    1. CLAUDE.md tabulaba el alert `deploy_lag_drift_vs_expected` como
       `Auto (implicit) + cron re-eval tras deploy completo o update de
       KV` pero NO documentaba el SOP humano: quién decide qué lado
       está rezagado, cuándo bumpear KV, cómo cerrar el alert. Sin SOP,
       un operador nuevo no sabe si el alert es benign o requiere
       redeploy urgente.
    2. Alerts con `alert_key` matching `scheduler_(missed|error)_<32hex>$`
       son one-off (job_id UUID único por invocación). NO recurren →
       nunca disparan EVENT_JOB_EXECUTED → el auto-resolve del listener
       (P1-NEW-2) no aplica. El sweep standard con TTL=24h los mantiene
       visibles 24h, pero tras 2-3h ya está claro que no recuperan.

Fix:
    1. Sub-sección "SOP: resolver `deploy_lag_drift_vs_expected`" en
       CLAUDE.md (6 pasos: identificar delta, decidir lado, bumpear KV,
       cerrar alert, verificar, post-mortem).
    2. Sweep adicional en `_resolve_stale_scheduler_alerts` con TTL
       agresivo (knob `MEALFIT_SCHEDULER_ALERT_TTL_ONEOFF_H` default 12h)
       filtrando por regex `^scheduler_(missed|error)_[0-9a-f]{32}$`.
       El tick observable (`_scheduler_alerts_sweep_tick`) extiende
       metadata con `swept_standard` y `swept_oneoff`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BACKEND_ROOT.parent
_CLAUDE_MD = _REPO_ROOT / "CLAUDE.md"
_CRON_PY = _BACKEND_ROOT / "cron_tasks.py"

# [Re-anchor 2026-06-12] Los pasos detallados del SOP (script SSOT, SQL de
# fallback, cierre manual del alert, casos expected>live / live>expected)
# se movieron de CLAUDE.md al runbook doc-first (patrón P3-CLAUDEMD-CAP):
# CLAUDE.md conserva el header + 1-line + link; el cuerpo operativo vive en
# `runbook_system_alerts_sops_2026_05_11.md` (memory dir del proyecto).
# El test ahora verifica la cadena completa: header en CLAUDE.md → link al
# runbook → literales operativos dentro del runbook. Mismo invariante:
# el operador que parte de CLAUDE.md llega a los comandos exactos.
_SOP_RUNBOOK_NAME = "runbook_system_alerts_sops_2026_05_11.md"
_HOME = Path(os.path.expanduser("~"))
_MEMORY_DIR_CANDIDATES = [
    _HOME / ".claude" / "projects"
    / "c--Users-angel-OneDrive-Escritorio-MealfitRD-IA" / "memory",
    _HOME / ".claude" / "projects"
    / "C--Users-angel-OneDrive-Escritorio-Nodalia-MealfitRD-Software-MealfitRD-IA"
    / "memory",
]


def _locate_sop_runbook() -> Path | None:
    for memory_dir in _MEMORY_DIR_CANDIDATES:
        candidate = memory_dir / _SOP_RUNBOOK_NAME
        if candidate.is_file():
            return candidate
    return None


def _extract_sop_block(src: str, heading_re: str) -> str:
    """Extrae el bloque del SOP desde el heading hasta el siguiente heading
    del mismo nivel o superior (`##`/`###`)."""
    m = re.search(
        heading_re + r"(.+?)(?=^###\s|^##\s|\Z)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# P3-A: SOP documentado en CLAUDE.md (header + link) → runbook (cuerpo)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def claude_md_src() -> str:
    return _CLAUDE_MD.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sop_body(claude_md_src: str) -> str:
    """Cuerpo operativo del SOP: bloque de CLAUDE.md + bloque del runbook.

    Falla (no skip) si el eslabón CLAUDE.md→runbook está roto: sin la
    referencia el operador nuevo no encuentra los comandos; sin el runbook
    el SOP completo desapareció."""
    claude_block = _extract_sop_block(
        claude_md_src,
        r"^###\s+SOP:\s+resolver\s+`deploy_lag_drift_vs_expected`",
    )
    assert claude_block, "Bloque del SOP no encontrado en CLAUDE.md (header presente pero sin cuerpo)."
    assert _SOP_RUNBOOK_NAME in claude_block, (
        "P3-CLEANUP regresión: el bloque del SOP en CLAUDE.md ya no "
        f"referencia `{_SOP_RUNBOOK_NAME}`. Tras el trim doc-first "
        "(P3-CLAUDEMD-CAP), el link es el único camino del operador a los "
        "pasos detallados — restaurarlo."
    )
    runbook_path = _locate_sop_runbook()
    assert runbook_path is not None, (
        f"Runbook `{_SOP_RUNBOOK_NAME}` ausente en los memory dirs "
        f"{[str(p) for p in _MEMORY_DIR_CANDIDATES]}. Sin él, el SOP "
        "referenciado desde CLAUDE.md apunta al vacío — restaurar desde "
        "memoria o re-crear desde el P-fix P3-CLEANUP."
    )
    runbook_src = runbook_path.read_text(encoding="utf-8")
    runbook_block = _extract_sop_block(
        runbook_src,
        r"^##\s+SOP:\s+resolver\s+`deploy_lag_drift_vs_expected`",
    )
    assert runbook_block, (
        f"Sección `## SOP: resolver \\`deploy_lag_drift_vs_expected\\`` "
        f"no encontrada dentro de {_SOP_RUNBOOK_NAME}."
    )
    return claude_block + "\n" + runbook_block


def test_sop_section_header_present(claude_md_src: str):
    """La sub-sección `### SOP: resolver \`deploy_lag_drift_vs_expected\``
    debe existir tras el SOP de `plan_data_corrupted` y antes de
    "Cómo añadir un nuevo alert_key"."""
    pattern = re.compile(
        r"^###\s+SOP:\s+resolver\s+`deploy_lag_drift_vs_expected`",
        re.MULTILINE,
    )
    assert pattern.search(claude_md_src), (
        "P3-CLEANUP regresión: sub-sección "
        "`### SOP: resolver \\`deploy_lag_drift_vs_expected\\`` "
        "ausente en CLAUDE.md. Sin SOP documentado, un operador nuevo "
        "no sabe qué hacer cuando el alert aparece."
    )


def test_sop_references_publish_pfix_marker_script(sop_body: str):
    """El SOP debe referenciar el script SSOT `publish_pfix_marker.py`
    como path de resolución preferido para bumpear el KV."""
    assert "publish_pfix_marker.py" in sop_body, (
        "P3-CLEANUP regresión: SOP no referencia el script SSOT "
        "`publish_pfix_marker.py`. Sin la referencia, el operador podría "
        "actualizar el KV manualmente sin saber que existe el script."
    )


def test_sop_provides_sql_update_kv_example(sop_body: str):
    """El SOP debe incluir el SQL UPDATE para actualizar el KV
    manualmente (fallback si el script falla o el operador no tiene
    `SUPABASE_DB_URL` seteado)."""
    assert "UPDATE app_kv_store" in sop_body, (
        "P3-CLEANUP regresión: SOP no incluye SQL UPDATE para el KV. "
        "El fallback manual es crítico — el script puede fallar por "
        "ausencia de SUPABASE_DB_URL en el shell del operador."
    )
    assert "expected_last_known_pfix" in sop_body, (
        "P3-CLEANUP regresión: SOP no menciona la key específica del KV "
        "(`expected_last_known_pfix`). Sin el nombre exacto, el "
        "operador podría actualizar otra fila."
    )


def test_sop_provides_close_alert_example(sop_body: str):
    """El SOP debe documentar cómo cerrar el alert manualmente si el
    cron tarda en re-evaluar (`UPDATE system_alerts SET resolved_at=NOW()`)."""
    assert "UPDATE system_alerts" in sop_body and "resolved_at" in sop_body, (
        "P3-CLEANUP regresión: SOP no documenta cómo cerrar el alert. "
        "Sin esto, el operador espera al cron periódico (24h default) "
        "incluso después de corregir el drift."
    )


def test_sop_distinguishes_live_vs_expected_lag(sop_body: str):
    """El SOP debe distinguir los dos casos: `expected > live` (binario
    rezagado, redeploy) vs `live > expected` (KV rezagado, bumpear KV)."""
    # Buscar mención de ambos casos (orden flexible).
    assert "expected > live" in sop_body or "expected vs live" in sop_body.lower(), (
        "P3-CLEANUP regresión: SOP no menciona el caso `expected > live` "
        "(binario rezagado). Operador no sabrá que debe redeployar."
    )
    assert "live > expected" in sop_body, (
        "P3-CLEANUP regresión: SOP no menciona el caso `live > expected` "
        "(KV rezagado). Operador no sabrá que debe bumpear el KV."
    )


# ---------------------------------------------------------------------------
# P3-B: Sweep one-off con TTL agresivo
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cron_src() -> str:
    return _CRON_PY.read_text(encoding="utf-8")


def _extract_function_body(src: str, name: str) -> str:
    m = re.search(
        rf"^(?:async\s+)?def\s+{re.escape(name)}\s*\([^)]*\)[^:]*:(.*?)(?=^(?:async\s+)?def\s)",
        src,
        re.DOTALL | re.MULTILINE,
    )
    assert m, f"No se encontró función top-level `{name}`."
    return m.group(1)


def test_resolve_sweep_includes_oneoff_path(cron_src: str):
    """`_resolve_stale_scheduler_alerts` debe incluir un sweep adicional
    filtrando por job_id UUID hex (regex `[0-9a-f]{32}`). Sin esto,
    alerts one-off conservan TTL standard (24h) pese a no recuperar."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    # Patrón regex POSIX en el SQL.
    assert "[0-9a-f]{32}" in body, (
        "P3-CLEANUP regresión: sweep one-off ausente en "
        "`_resolve_stale_scheduler_alerts`. El regex POSIX "
        "`^scheduler_(missed|error)_[0-9a-f]{32}$` distingue UUIDs hex "
        "one-off (jobs que nunca recurren) del scheduler estándar."
    )


def test_oneoff_sweep_uses_dedicated_ttl_knob(cron_src: str):
    """El sweep one-off debe leer el knob `MEALFIT_SCHEDULER_ALERT_TTL_ONEOFF_H`
    (default 12h, más agresivo que el TTL estándar 24h)."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    assert "MEALFIT_SCHEDULER_ALERT_TTL_ONEOFF_H" in body, (
        "P3-CLEANUP regresión: sweep one-off NO usa knob dedicado. "
        "Sin knob, no hay forma de ajustar TTL en runtime — usar "
        "12h hardcoded sería rígido."
    )
    # Default 12 debe aparecer como número entero junto al knob.
    pattern = re.compile(
        r'_env_int\(\s*["\']MEALFIT_SCHEDULER_ALERT_TTL_ONEOFF_H["\']\s*,\s*12\s*\)'
    )
    assert pattern.search(body), (
        "P3-CLEANUP regresión: knob `MEALFIT_SCHEDULER_ALERT_TTL_ONEOFF_H` "
        "sin default 12. Default debe ser <24 (TTL standard) para que "
        "el sweep one-off sea MÁS agresivo que el estándar."
    )


def test_oneoff_sweep_inside_try_except(cron_src: str):
    """El sweep one-off debe estar wrapped en try/except — un fallo
    NO debe abortar el resto del flujo (sweep standard ya pasó,
    tick observable debe seguir emitiéndose).

    Estrategia: encontrar el `try:` MÁS CERCANO antes del regex hex
    (debe ser el del bloque del sweep one-off, no de otro upstream)
    y verificar que un `except` aparece después del hex y antes del
    siguiente `try:` (delimita el bloque del one-off)."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    hex_idx = body.find("[0-9a-f]{32}")
    assert hex_idx >= 0, "regex hex one-off no encontrado"

    # `try:` cercano antes del hex.
    before = body[:hex_idx]
    last_try_idx = before.rfind("try:")
    assert last_try_idx >= 0, (
        "P3-CLEANUP regresión: no hay `try:` antes del sweep one-off "
        "en `_resolve_stale_scheduler_alerts`. El sweep debe estar "
        "wrapped — sino una excepción aborta el tick observable."
    )
    # Verificar que no hay un `except` entre el try y el hex (sería un
    # try: de otro bloque ya cerrado).
    between_try_and_hex = body[last_try_idx:hex_idx]
    assert "except" not in between_try_and_hex, (
        "P3-CLEANUP regresión: el `try:` encontrado antes del hex pertenece "
        "a otro bloque (su `except` aparece antes del hex). Sweep one-off "
        "necesita su propio try/except."
    )
    # `except` después del hex, antes del siguiente `try:` o EOF.
    after = body[hex_idx:]
    next_try_idx = after.find("try:")
    end_idx = next_try_idx if next_try_idx > 0 else len(after)
    section = after[:end_idx]
    assert "except" in section, (
        "P3-CLEANUP regresión: no hay `except` que cierre el `try:` del "
        "sweep one-off. Best-effort: una excepción aquí NO debe abortar "
        "el tick observable que viene después (P2-B-OBS)."
    )


def test_tick_metadata_extended_with_oneoff_fields(cron_src: str):
    """El tick observable (`_scheduler_alerts_sweep_tick`) debe extender
    su metadata con campos one-off (`swept_oneoff`, `ttl_oneoff_hours`)
    para que el post-mortem correlacione ambos paths."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    tick_idx = body.find("_scheduler_alerts_sweep_tick")
    assert tick_idx >= 0, "tick observable no encontrado (cubierto por P2-B-OBS)"
    # Buscar campos one-off en el metadata cercano al tick.
    tick_block = body[tick_idx:tick_idx + 1000]
    assert "swept_oneoff" in tick_block, (
        "P3-CLEANUP regresión: tick metadata sin `swept_oneoff`. "
        "Post-mortem no puede separar count one-off del count standard."
    )
    assert "ttl_oneoff_hours" in tick_block, (
        "P3-CLEANUP regresión: tick metadata sin `ttl_oneoff_hours`. "
        "Operador no puede correlacionar swept_oneoff con el knob actual."
    )


def test_tick_metadata_preserves_swept_count_total(cron_src: str):
    """El campo `swept_count` del metadata debe ser TOTAL (standard +
    one-off), NO solo standard. Defensa contra dashboards que esperaban
    el comportamiento P2-B-OBS original."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    tick_idx = body.find("_scheduler_alerts_sweep_tick")
    tick_block = body[tick_idx:tick_idx + 1000]
    # `swept_count` debe sumar ambos (busca patrón aritmético).
    assert re.search(
        r'"swept_count"\s*:\s*\w+\s*\+\s*\w+',
        tick_block,
    ), (
        "P3-CLEANUP regresión: `swept_count` del tick no es suma "
        "standard + one-off. Backward-compat: el campo `swept_count` "
        "se mantuvo como TOTAL para no romper dashboards downstream "
        "que ya consultan esa key."
    )


def test_knob_default_strictly_less_than_standard_ttl(cron_src: str):
    """Sanity: TTL one-off (default 12h) debe ser ESTRICTAMENTE menor
    que TTL standard (default 24h). Si one-off > standard, el sweep
    estándar barre todo antes y el one-off nunca encuentra nada."""
    body = _extract_function_body(cron_src, "_resolve_stale_scheduler_alerts")
    # Extraer defaults de ambos knobs.
    standard = re.search(
        r'_env_int\(\s*["\']MEALFIT_SCHEDULER_ALERT_TTL_H["\']\s*,\s*(\d+)\s*\)',
        body,
    )
    oneoff = re.search(
        r'_env_int\(\s*["\']MEALFIT_SCHEDULER_ALERT_TTL_ONEOFF_H["\']\s*,\s*(\d+)\s*\)',
        body,
    )
    assert standard and oneoff, "Knobs TTL no detectados (cubierto por otros tests)."
    standard_h = int(standard.group(1))
    oneoff_h = int(oneoff.group(1))
    assert oneoff_h < standard_h, (
        f"P3-CLEANUP regresión: default TTL one-off ({oneoff_h}h) >= "
        f"default TTL standard ({standard_h}h). Si one-off >= standard, "
        f"el sweep one-off es no-op (estándar barre todo primero). "
        f"Intención del P-fix: one-off DEBE ser más agresivo (< 24h)."
    )
