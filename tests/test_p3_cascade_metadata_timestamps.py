"""[P3-CASCADE-METADATA · 2026-05-12] El detector cascade enriquece la
metadata del alert `scheduler_cascade_missed` con `last_missed_at_per_job`.

Pre-fix:
    `metadata` exponía solo `top_jobs_by_count: {<job>: <count>}` (counts
    agregados sin eje temporal). Ejemplo real audit 2026-05-12:
        {"top_jobs_by_count": {"flush_pending_deferrals": 1,
                              "alert_chunk_dual_processing": 1,
                              "flush_pending_lesson_telemetry": 1}}

    SRE recibía esa señal y NO podía decidir cuál job priorizar para root
    cause: ¿está fallando ACTIVAMENTE o es residuo del lookback (1h)? Para
    saberlo había que abrir logs separados o queries SQL adicionales contra
    `system_alerts` filtrando por cada `scheduler_missed_<job>` individual.

Fix:
    Trackear `last_missed_at_per_job: {<job>: <iso8601>}` durante el loop
    de conteo. Como la query del SELECT viene `ORDER BY triggered_at DESC`,
    el PRIMER occurrence de cada job_id es su missed más reciente (solo
    registramos si no estaba ya en el dict).

Beneficio operacional:
    SRE compara `last_missed_at_per_job` contra NOW():
      - Job con `last_seen ≈ NOW()` → fallando AHORA, prioridad #1.
      - Job con `last_seen ≈ NOW() - 50min` → residuo del lookback, prob.
        ya estabilizó, prioridad #N.

Persistencia:
    El nuevo dict se incluye en 2 lugares:
      1. UPSERT del alert `scheduler_cascade_missed` (visible en dashboard).
      2. `pipeline_metrics._scheduler_cascade_autoheal` (post-mortem).

Test parser-based: anchor + variable populada + ambas persistencias incluyen
la key + timestamps son ISO 8601 normalizados (no datetime raw).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON_TASKS = _BACKEND / "cron_tasks.py"


@pytest.fixture(scope="module")
def cron_text() -> str:
    return _CRON_TASKS.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Anchor presente — load-bearing para cross-link P2-HIST-AUDIT-14
# ---------------------------------------------------------------------------
def test_anchor_present(cron_text):
    assert "P3-CASCADE-METADATA" in cron_text, (
        "Anchor `P3-CASCADE-METADATA` removido — el cross-link del marker "
        "y este test dependen del anchor para localizar el bloque."
    )


# ---------------------------------------------------------------------------
# 2. La variable se construye dentro de _alert_scheduler_cascade_missed
# ---------------------------------------------------------------------------
def _extract_detector_block(text: str) -> str:
    """Aísla el cuerpo de `_alert_scheduler_cascade_missed` desde su `def`
    hasta el siguiente `def` a column 0 o EOF."""
    m = re.search(
        r"def _alert_scheduler_cascade_missed\(\):.*?(?=^def\s+\w+\s*\(|\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m, (
        "`def _alert_scheduler_cascade_missed():` no encontrado en cron_tasks.py."
    )
    return m.group(0)


def test_last_missed_at_per_job_dict_initialized(cron_text):
    """Dict `last_missed_at_per_job` debe existir como `dict[str, str]`
    inicializado vacío antes del loop sobre `rows`."""
    block = _extract_detector_block(cron_text)
    assert "last_missed_at_per_job: dict[str, str] = {}" in block, (
        "Variable `last_missed_at_per_job` no inicializada en el detector. "
        "Sin esto, el dict no existe → key ausente del UPSERT → alert "
        "regresa al estado pre-P3 (counts sin eje temporal)."
    )


def test_loop_populates_only_first_occurrence(cron_text):
    """Como el SELECT viene `ORDER BY triggered_at DESC`, el PRIMER
    occurrence de cada job_id es el más reciente. La asignación debe
    estar gateada por `if job_id not in last_missed_at_per_job:`."""
    block = _extract_detector_block(cron_text)
    assert re.search(
        r"if\s+job_id\s+not\s+in\s+last_missed_at_per_job\s*:",
        block,
    ), (
        "Falta el guard `if job_id not in last_missed_at_per_job:` antes "
        "de la asignación. Sin él, el último row del loop (el MÁS VIEJO) "
        "sobrescribe al MÁS RECIENTE — semántica invertida silenciosa."
    )


def test_isoformat_normalization(cron_text):
    """Los timestamps deben normalizarse a ISO 8601 (`triggered_at.isoformat()`)
    para que el JSON sea determinista. `datetime` raw rompería el JSON
    serializer y/o produciría formato variable según driver."""
    block = _extract_detector_block(cron_text)
    assert "triggered_at.isoformat()" in block, (
        "Falta `triggered_at.isoformat()` — sin normalización el JSON serializer "
        "puede romper o producir formato variable según driver psycopg/asyncpg."
    )


def test_select_orders_by_triggered_at_desc(cron_text):
    """La invariante `primer occurrence wins` depende de `ORDER BY
    triggered_at DESC` en el SELECT. Si alguien lo cambia a ASC sin
    actualizar este loop, el dict registraría el más VIEJO por error."""
    block = _extract_detector_block(cron_text)
    assert "ORDER BY triggered_at DESC" in block, (
        "El SELECT debe ordenar por `triggered_at DESC` para que el primer "
        "match en el loop sea el más reciente. Si necesitas cambiar a ASC, "
        "invierte la lógica del guard `if job_id not in ...:` o el dict "
        "registrará el más viejo silenciosamente."
    )


# ---------------------------------------------------------------------------
# 3. Persistencia en AMBOS sitios: UPSERT alert + pipeline_metrics autoheal
# ---------------------------------------------------------------------------
def test_alert_upsert_includes_last_missed_at_per_job(cron_text):
    """El UPSERT a `system_alerts` con alert_key `scheduler_cascade_missed`
    debe incluir `last_missed_at_per_job` en el `metadata` jsonb."""
    block = _extract_detector_block(cron_text)
    # Localizar el bloque del UPSERT (entre `INSERT INTO system_alerts` y
    # el `logger.error("🚨 ...")`).
    m = re.search(
        r"INSERT INTO system_alerts.*?logger\.error",
        block,
        re.DOTALL,
    )
    assert m, "Bloque del UPSERT a system_alerts no localizable."
    upsert_block = m.group(0)
    assert '"last_missed_at_per_job": last_missed_at_per_job' in upsert_block, (
        "El UPSERT del alert `scheduler_cascade_missed` NO incluye "
        "`last_missed_at_per_job` en metadata. Sin esta key, dashboard SRE "
        "regresa al estado pre-P3: counts agregados sin eje temporal → "
        "imposible priorizar root cause sin abrir logs separados."
    )


def test_autoheal_pipeline_metric_includes_last_missed_at_per_job(cron_text):
    """El segundo INSERT (a `pipeline_metrics._scheduler_cascade_autoheal`)
    debe incluir la misma key — permite post-mortem correlacionar qué jobs
    estaban activos vs residuo cuando la cascada fue detectada."""
    block = _extract_detector_block(cron_text)
    # El INSERT al autoheal viene después del UPSERT del alert.
    m = re.search(
        r'"_scheduler_cascade_autoheal".*?json\.dumps\(\{.*?\}.*?\)',
        block,
        re.DOTALL,
    )
    assert m, "Bloque del INSERT autoheal no localizable."
    autoheal_block = m.group(0)
    assert '"last_missed_at_per_job": last_missed_at_per_job' in autoheal_block, (
        "`pipeline_metrics._scheduler_cascade_autoheal` NO incluye "
        "`last_missed_at_per_job`. La key debe duplicarse en ambos sitios "
        "para que post-mortem pueda correlacionar (UPSERT alert se cierra "
        "tras minutos; pipeline_metrics persiste para análisis offline)."
    )


# ---------------------------------------------------------------------------
# 4. Comment del rationale presente — load-bearing para futuros readers
# ---------------------------------------------------------------------------
def test_rationale_comment_present(cron_text):
    """El comment que documenta la semántica `primer occurrence wins`
    + razón del eje temporal debe permanecer en el bloque del loop."""
    block = _extract_detector_block(cron_text)
    assert "P3-CASCADE-METADATA" in block, (
        "Comment con anchor `P3-CASCADE-METADATA` removido del bloque del "
        "loop. Sin él, futuro reader puede 'optimizar' el guard `if not in` "
        "pensando que es redundante (rompiendo la semántica más-reciente-wins)."
    )
