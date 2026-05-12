"""[P3-NEXT-4 · 2026-05-11] El cron horario
`_aggregate_coherence_block_history_metrics` (P3-B) DEBE contar las
4 nuevas surfaces de P1-NEXT-2 + P2-NEXT-2 en buckets dedicados, en
lugar de descartarlas via else-branch (`logger.debug + ignore`).

Cierra el gap residual del audit 2026-05-11:
    P1-NEXT-2 y P2-NEXT-2 introdujeron 4 nuevos `action_taken` values
    en `_shopping_coherence_block_history`:
      - `warn_only_chunk_t2`     (_chunk_worker T2)
      - `warn_only_recalc`       (/recalculate-shopping-list)
      - `warn_only_agent_tool`   (tools.modify_single_meal)
      - `warn_only_cron_daily`   (cron diario 04:00 UTC, P2-NEXT-2)

    Antes de P3-NEXT-4, el aggregator solo conocía `not_applicable`,
    `degrade`, `reject_minor`, `reject_high`, `hydration_error`,
    `post_swap_revalidation`. Los 4 nuevos values caían al else-branch
    (`logger.debug + ignore`) → cron infrarreportaba volumen real de
    surfaces de coherencia.

Fix P3-NEXT-4:
    - 4 nuevos buckets en `counts` dict.
    - `surface_breakdown` derivada para dashboards/alertas (sin
      sumar buckets manualmente en cada query).
    - Pipeline_metrics metadata incluye breakdown.
    - Sin nuevos knobs (sigue todo bajo MEALFIT_COHERENCE_METRICS_*).

Drift detection:
    - Un bucket nuevo borrado → falla.
    - `surface_breakdown` se rompe (clave faltante) → falla.
    - pipeline_metrics metadata pierde `surface_breakdown` → falla.

Tooltip-anchor: P3-NEXT-4-START | gap audit 2026-05-11
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


_BACKEND = Path(__file__).resolve().parent.parent
_CRON = _BACKEND / "cron_tasks.py"


def _read_function_body(source: str, fn_name: str) -> str:
    pattern = re.compile(
        rf"^def\s+{re.escape(fn_name)}\s*\(",
        re.MULTILINE,
    )
    m = pattern.search(source)
    if not m:
        return ""
    next_def_pattern = re.compile(r"^(def |class |@)", re.MULTILINE)
    next_def = next_def_pattern.search(source, pos=m.end())
    if next_def:
        return source[m.start():next_def.start()]
    return source[m.start():]


@pytest.fixture(scope="module")
def cron_body() -> str:
    source = _CRON.read_text(encoding="utf-8")
    body = _read_function_body(source, "_aggregate_coherence_block_history_metrics")
    assert body, (
        "Función `_aggregate_coherence_block_history_metrics` no encontrada. "
        "El test P3-NEXT-4 perdió su anchor."
    )
    return body


# ---------------------------------------------------------------------------
# 1. Los 4 buckets nuevos están en el dict `counts`
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("bucket_key", [
    "warn_only_chunk_t2",
    "warn_only_recalc",
    "warn_only_agent_tool",
    "warn_only_cron_daily",
])
def test_bucket_present_in_counts(cron_body: str, bucket_key: str):
    """Cada bucket de P1-NEXT-2 + P2-NEXT-2 debe existir en `counts`."""
    pattern = re.compile(
        rf"['\"]" + re.escape(bucket_key) + r"['\"]\s*:\s*0",
    )
    assert pattern.search(cron_body), (
        f"P3-NEXT-4 violation: bucket `{bucket_key}` ausente en `counts` dict "
        f"del aggregator. Sin él, entries con "
        f"`action_taken='{bucket_key}'` caen al else-branch (log debug + "
        f"ignore) → cron infrarreporta volumen real."
    )


# ---------------------------------------------------------------------------
# 2. surface_breakdown está construido y referencia los 4 buckets
# ---------------------------------------------------------------------------
def test_surface_breakdown_constructed(cron_body: str):
    """`surface_breakdown` debe construirse a partir de los counts y
    referenciar los 4 buckets nuevos para que dashboards/alertas no
    tengan que sumar manualmente."""
    assert "surface_breakdown" in cron_body, (
        "P3-NEXT-4 violation: variable `surface_breakdown` no construida. "
        "Dashboards quedan obligados a sumar buckets individuales para "
        "ver volumen por surface."
    )


@pytest.mark.parametrize("surface_key,bucket_ref", [
    ("chunked_t2",  "warn_only_chunk_t2"),
    ("recalc",      "warn_only_recalc"),
    ("agent_tool",  "warn_only_agent_tool"),
    ("cron_daily",  "warn_only_cron_daily"),
])
def test_surface_breakdown_references_bucket(cron_body: str, surface_key: str, bucket_ref: str):
    """Cada surface en `surface_breakdown` debe referenciar el bucket
    correspondiente de `counts`."""
    # Buscar patrón "<surface_key>": ... counts["<bucket_ref>"]
    # Permisivo: la línea puede tener whitespace/comentarios.
    pattern = re.compile(
        rf"['\"]" + re.escape(surface_key) + r"['\"]\s*:\s*[^,}}]*counts\[\s*['\"]" + re.escape(bucket_ref) + r"['\"]",
    )
    assert pattern.search(cron_body), (
        f"P3-NEXT-4 violation: `surface_breakdown['{surface_key}']` "
        f"no referencia `counts['{bucket_ref}']`. Sin este link, el "
        f"breakdown queda desincronizado de los counts y dashboard "
        f"reportará 0 para esa surface aunque el bucket tenga entries."
    )


# ---------------------------------------------------------------------------
# 3. pipeline_metrics metadata incluye surface_breakdown
# ---------------------------------------------------------------------------
def test_pipeline_metrics_metadata_has_surface_breakdown(cron_body: str):
    """El INSERT a `pipeline_metrics` debe persistir `surface_breakdown`
    en metadata (no solo en logs). Sin esto, dashboards históricos no
    pueden hacer query SQL al breakdown."""
    # Buscar "surface_breakdown": surface_breakdown dentro del INSERT block.
    pattern = re.compile(
        r"['\"]surface_breakdown['\"]\s*:\s*surface_breakdown",
    )
    assert pattern.search(cron_body), (
        "P3-NEXT-4 violation: `surface_breakdown` NO se persiste en "
        "`pipeline_metrics.metadata`. Aunque el cron logguea el dict, "
        "queries SQL históricas (Grafana, post-mortem) no pueden "
        "acceder. Fix: añadir `\"surface_breakdown\": surface_breakdown` "
        "al dict metadata del INSERT a pipeline_metrics."
    )


# ---------------------------------------------------------------------------
# 4. P3-NEXT-4 marker presente en código (forensics)
# ---------------------------------------------------------------------------
def test_p3_next_4_marker_present(cron_body: str):
    """Marker P3-NEXT-4 debe aparecer en al menos un comentario para
    forensics — un operador que vea el bucket nuevo en producción
    debe poder grep el marker y encontrar la motivación."""
    assert "P3-NEXT-4" in cron_body, (
        "P3-NEXT-4 violation: marker `P3-NEXT-4` ausente en el cuerpo "
        "del aggregator. Sin marker, forensics post-deploy es ciego."
    )


# ---------------------------------------------------------------------------
# 5. Cross-link slug
# ---------------------------------------------------------------------------
def test_marker_anchor_present():
    expected_slug = "p3_next_4"
    assert expected_slug in __file__.replace("\\", "/").lower(), (
        "Filename debe contener slug `p3_next_4` para cross-link."
    )
